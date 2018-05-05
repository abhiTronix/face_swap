#include "face_swap/face_swap.h"
#include "face_swap/utilities.h"

// std
#include <limits>

// OpenCV
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // Debug

namespace face_swap
{
    FaceSwap::FaceSwap(const std::string& landmarks_path, const std::string& model_3dmm_h5_path,
        const std::string& model_3dmm_dat_path, const std::string& reg_model_path,
        const std::string& reg_deploy_path, const std::string& reg_mean_path,
        bool generic, bool with_expr, bool with_gpu, int gpu_device_id) :
		m_with_gpu(with_gpu),
		m_gpu_device_id(gpu_device_id)
    {
        // Initialize Sequence Face Landmarks
        m_sfl = sfl::SequenceFaceLandmarks::create(landmarks_path);

        // Initialize CNN 3DMM with exression
        m_cnn_3dmm_expr = std::make_unique<CNN3DMMExpr>(
			reg_deploy_path, reg_model_path, reg_mean_path, model_3dmm_dat_path,
			generic, with_expr, with_gpu, gpu_device_id);

        // Load Basel 3DMM
        m_basel_3dmm = std::make_unique<Basel3DMM>();
        *m_basel_3dmm = Basel3DMM::load(model_3dmm_h5_path);

        // Create renderer
        m_face_renderer = std::make_unique<FaceRenderer>();
    }

	void FaceSwap::setSegmentationModel(const std::string& seg_model_path,
		const std::string& seg_deploy_path)
	{
		m_face_seg = std::make_unique<face_seg::FaceSeg>(seg_deploy_path,
			seg_model_path, m_with_gpu, m_gpu_device_id);
	}

	void FaceSwap::clearSegmentationModel()
	{
		m_face_seg = nullptr;
	}

	bool FaceSwap::isSegmentationModelInit()
	{
		return m_face_seg != nullptr;
	}

	bool FaceSwap::setSource(const cv::Mat& img, const cv::Mat& seg)
    {
        m_source_img = img;

        // Preprocess image
        std::vector<cv::Point> cropped_landmarks;
        cv::Mat cropped_img, cropped_seg;
        if (!preprocessImages(img, seg, m_src_landmarks, cropped_landmarks,
			cropped_img, cropped_seg))
            return false;

		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_seg.empty() && m_face_seg != nullptr)
			cropped_seg = m_face_seg->process(cropped_img);

        // Calculate coefficients and pose
        cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
        cv::Mat vecR, vecT, K;
        m_cnn_3dmm_expr->process(cropped_img, cropped_landmarks, shape_coefficients,
            tex_coefficients, expr_coefficients, vecR, vecT, K);

        // Create mesh
        m_src_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients, 
            expr_coefficients);

        // Texture mesh
        generateTexture(m_src_mesh, cropped_img, cropped_seg, vecR, vecT, K, 
            m_tex, m_uv);

        /// Debug ///
        m_src_cropped_img = cropped_img;
        m_src_cropped_seg = cropped_seg;
        m_src_cropped_landmarks = cropped_landmarks;
        m_src_vecR = vecR;
        m_src_vecT = vecT;
        m_src_K = K;
        /////////////

        return true;
    }

    bool FaceSwap::setTarget(const cv::Mat& img, const cv::Mat& seg)
    {
        m_target_img = img;
        m_target_seg = seg;

        // Preprocess image
        std::vector<cv::Point> cropped_landmarks;
        cv::Mat cropped_img, cropped_seg;
        if (!preprocessImages(img, seg, m_tgt_landmarks, cropped_landmarks,
            cropped_img, cropped_seg, m_target_bbox))
            return false;

		// If segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_seg.empty() && m_face_seg != nullptr)
		{
			cropped_seg = m_face_seg->process(cropped_img);
			m_target_seg = cv::Mat::zeros(img.size(), CV_8U);
			cropped_seg.copyTo(m_target_seg(m_target_bbox));
		}
			
        m_tgt_cropped_img = cropped_img;
        m_tgt_cropped_seg = cropped_seg;
        
        /// Debug ///
        m_tgt_cropped_landmarks = cropped_landmarks;
        /////////////

        // Calculate coefficients and pose
        cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
        m_cnn_3dmm_expr->process(cropped_img, cropped_landmarks, shape_coefficients,
            tex_coefficients, expr_coefficients, m_vecR, m_vecT, m_K);

        // Create mesh
        m_dst_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients,
            expr_coefficients);
        m_dst_mesh.tex = m_tex;
        m_dst_mesh.uv = m_uv;

        // Initialize renderer
        m_face_renderer->init(cropped_img.cols, cropped_img.rows);
        m_face_renderer->setProjection(m_K.at<float>(4));
        m_face_renderer->setMesh(m_dst_mesh);

        return true;
    }

	void horFlipLandmarks(std::vector<cv::Point>& landmarks, int width)
	{
		// Invert X coordinates
		for (cv::Point& p : landmarks)
			p.x = width - p.x;

		// Jaw
		for (int i = 0; i <= 7; ++i)	
			std::swap(landmarks[i], landmarks[16 - i]);

		// Eyebrows
		for (int i = 17; i <= 21; ++i)	
			std::swap(landmarks[i], landmarks[43 - i]);

		// Nose
		std::swap(landmarks[31], landmarks[35]);
		std::swap(landmarks[32], landmarks[34]);

		// Eyes
		std::swap(landmarks[36], landmarks[45]);
		std::swap(landmarks[37], landmarks[44]);
		std::swap(landmarks[38], landmarks[43]);
		std::swap(landmarks[39], landmarks[42]);
		std::swap(landmarks[40], landmarks[47]);
		std::swap(landmarks[41], landmarks[46]);

		// Mouth Outer
		std::swap(landmarks[48], landmarks[54]);
		std::swap(landmarks[49], landmarks[53]);
		std::swap(landmarks[50], landmarks[52]);
		std::swap(landmarks[59], landmarks[55]);
		std::swap(landmarks[58], landmarks[56]);

		// Mouth inner
		std::swap(landmarks[60], landmarks[64]);
		std::swap(landmarks[61], landmarks[63]);
		std::swap(landmarks[67], landmarks[65]);
	}

    void draw_green_pixel(
        const std::string& input_image,
        const std::string& output_image,
        int row, int col)
    {
        cv::Mat img = cv::imread(input_image);

        for (int r = row - 1; r <= row + 1; r++)
        {
            for (int c = col - 1; c <= col + 1; c++)
            {
                // the loop below assumes that the image
                // is a 8-bit 3-channel. check it.
                CV_Assert(img.type() == CV_8UC3);

                uchar* data = img.at<unsigned char[3]>(r, c);
                //enum { BLUE, GREEN, RED };
                *data++ = 0;
                *data++ = 255;
                *data++ = 0;
            }

        }
        cv::imwrite(output_image, img);
    }

	bool FaceSwap::setImages(const cv::Mat& src, const cv::Mat& tgt,
		const cv::Mat& src_seg, const cv::Mat& tgt_seg)
	{
		m_source_img = src;
		m_target_img = tgt;
		m_target_seg = tgt_seg;
#if 0
        m_src_to_dst = cv::Mat(m_source_img.size(), CV_16UC3, 0); /* std::numeric_limits<unsigned short>::max()); */
        for (int r = 0; r < m_src_to_dst.rows; r++) {
            for (int c = 0; c < m_src_to_dst.cols; c++) {
                auto elem = m_src_to_dst.at<unsigned short[3]>(r, c);
                elem[0] = r; elem[1] = c;
            }
        }
#endif           

		// Preprocess source image
		std::vector<cv::Point> cropped_src_landmarks;
		cv::Mat cropped_src, cropped_src_seg;
		if (!preprocessImages(src, src_seg, m_src_landmarks, cropped_src_landmarks,
			cropped_src, cropped_src_seg, m_source_bbox))
			return false;

        cv::imwrite("C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\cropped_src.png", cropped_src);

		// Preprocess target image
		std::vector<cv::Point> cropped_tgt_landmarks;
		cv::Mat cropped_tgt, cropped_tgt_seg;
		if (!preprocessImages(tgt, tgt_seg, m_tgt_landmarks, cropped_tgt_landmarks,
			cropped_tgt, cropped_tgt_seg, m_target_bbox))
			return false;

        cv::imwrite("C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\cropped_tgt.png", cropped_tgt);
        
        // Check if horizontal flip is required
		float src_angle = sfl::getFaceApproxHorAngle(cropped_src_landmarks);
		float tgt_angle = sfl::getFaceApproxHorAngle(cropped_tgt_landmarks);
		if ((src_angle * tgt_angle) < 0 && std::abs(src_angle - tgt_angle) > (CV_PI / 18.0f))
		{
			// Horizontal flip the source image
			cv::flip(cropped_src, cropped_src, 1);
			if(!cropped_src_seg.empty())
				cv::flip(cropped_src_seg, cropped_src_seg, 1);

			// Horizontal flip the source landmarks
			horFlipLandmarks(cropped_src_landmarks, cropped_src.cols);

            // Flip pixels in m_src_to_dst
            //cv::flip(m_src_to_dst(m_source_bbox), m_src_to_dst(m_source_bbox), 1);
            //auto cropped_pixels = m_src_to_dst(m_source_bbox);
            //for (int r = 0; r < cropped_pixels.rows; r++) {
            //    for (int c = 0; c < cropped_pixels.cols; c++) {
            //        auto elem = cropped_pixels.at<unsigned short[3]>(r, c);
            //        elem[2] = 1;
            //    }
            //}

		}

		// If source segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_src_seg.empty() && m_face_seg != nullptr)
			cropped_src_seg = m_face_seg->process(cropped_src);

		// If target segmentation was not specified and we have a segmentation model then
		// calculate the segmentation
		if (cropped_tgt_seg.empty() && m_face_seg != nullptr)
		{
			cropped_tgt_seg = m_face_seg->process(cropped_tgt);
			m_target_seg = cv::Mat::zeros(tgt.size(), CV_8U);
			cropped_tgt_seg.copyTo(m_target_seg(m_target_bbox));
		}

		m_tgt_cropped_img = cropped_tgt;
		m_tgt_cropped_seg = cropped_tgt_seg;

		/// Debug ///
		m_tgt_cropped_landmarks = cropped_tgt_landmarks;
		/////////////

		// Calculate source coefficients and pose
		{
			cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
			cv::Mat vecR, vecT, K;
			m_cnn_3dmm_expr->process(cropped_src, cropped_src_landmarks, shape_coefficients,
				tex_coefficients, expr_coefficients, vecR, vecT, K);

			// Create source mesh
			m_src_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients,
				expr_coefficients);

			// Texture source mesh
			generateTexture(m_src_mesh, cropped_src, cropped_src_seg, vecR, vecT, K,
				m_tex, m_uv);

            boost::filesystem::path out_file_path = "c:\\face_swap\\img_cropped.jpg";
            cv::imwrite(out_file_path.string(), cropped_src);

			/// Debug ///
			m_src_cropped_img = cropped_src;
			m_src_cropped_seg = cropped_src_seg;
			m_src_cropped_landmarks = cropped_src_landmarks;
			m_src_vecR = vecR;
			m_src_vecT = vecT;
			m_src_K = K;
			/////////////
		}
		
		// Calculate target coefficients and pose
        {
            cv::Mat shape_coefficients, tex_coefficients, expr_coefficients;
            m_cnn_3dmm_expr->process(cropped_tgt, cropped_tgt_landmarks, shape_coefficients,
                tex_coefficients, expr_coefficients, m_vecR, m_vecT, m_K);

            // Create target mesh
            m_dst_mesh = m_basel_3dmm->sample(shape_coefficients, tex_coefficients,
                expr_coefficients);

            //int row = 210;
            //int col = 165;

            //CV_Assert(m_tex.type() == CV_8UC4);
            //m_tex = cv::Mat::zeros(m_tex.size(), m_tex.type());
            //CV_Assert(m_tex.type() == CV_8UC4);
            //unsigned char* data = m_tex.at<unsigned char[4]>(row, col);
            //*data++ = 255;
            //*data++ = 255;
            //*data++ = 255;
            //*data++ = 255;

            //draw_green_pixel(
            //    "C:\\face_swap\\pics\\brad.jpg",
            //    "C:\\face_swap\\pics\\brad_green.jpg",
            //    row,
            //    col);

            m_dst_mesh.tex = m_tex;
            m_dst_mesh.uv = m_uv;

        }

		m_face_renderer->init(cropped_tgt.cols, cropped_tgt.rows);
		m_face_renderer->setProjection(m_K.at<float>(4));
        generateSourceMappings();
#if 0
        generateMappings();
        generateAbsoluteMappings();
        dumpMappings();
        dumpAbsoluteMappings();
        
        // Apply the absolute mapping on src img: W(I1).
        //applyMapping();

        cv::Mat pow_img_rgb;
        cv::Mat diff_img_rgb_float;
        cv::Mat diff_img_rgb = cv::max(m_target_img, m_W_i1) - cv::min(m_target_img, m_W_i1);
        std::cout << diff_img_rgb.type() << std::endl;
        diff_img_rgb = diff_img_rgb.mul(m_W_mask);
        diff_img_rgb.convertTo(diff_img_rgb_float, CV_32F);
        cv::pow(diff_img_rgb_float, 2, pow_img_rgb);
        m_score = cv::mean(pow_img_rgb);
        
        cv::Mat diff_img_grayscale;
        cv::cvtColor(diff_img_rgb, diff_img_grayscale, cv::COLOR_RGB2GRAY);
        std::string diff_img_grayscale_path = "diff_img_grayscale.jpg";
        std::string diff_img_rgb_path = "diff_img_rgb.jpg";
        cv::imwrite(diff_img_grayscale_path, diff_img_grayscale);
        cv::imwrite(diff_img_rgb_path, diff_img_rgb);

        dumpStats();
#endif
        /*paintMappings();*/

        // Initialize renderer
        m_dst_mesh.tex = m_tex;
        m_face_renderer->setMesh(m_dst_mesh);


		return true;
	}

    void FaceSwap::applyMapping()
    {
        CV_Assert(m_source_img.type() == CV_8UC3);
        //m_W_i1 = m_source_img.clone();
        m_W_i1 = cv::Mat::zeros(m_source_img.size(), m_source_img.type());
        m_W_mask = cv::Mat::zeros(m_source_img.size(), m_source_img.type());
        const uchar mask_vals[] = {1, 1, 1};
        for (const auto& pointsPair : m_absolutePixelMappings)
        {
            const uchar* src_pixel = m_source_img.at<unsigned char[3]>(pointsPair.first.x, pointsPair.first.y);
            for (const auto& targetPoints : pointsPair.second)
            {
                uchar* dst_pixel = m_W_i1.at<unsigned char[3]>(targetPoints.x, targetPoints.y);
                uchar* dst_mask_pixel = m_W_mask.at<unsigned char[3]>(targetPoints.x, targetPoints.y);
                // TODO: there must be a better way of copying pixels!
                memcpy(dst_pixel, src_pixel, sizeof(unsigned char[3]));
                
                // Build target mask
                memcpy(dst_mask_pixel, mask_vals, sizeof(mask_vals));

                //pow(diff_image_rgb .* dst_mask_pixel);
            }
        }

        std::string w_i1_path = "w_i1.jpg";
        cv::imwrite(w_i1_path, m_W_i1);

        auto diff_image = m_W_i1(m_source_bbox) - m_source_img(m_source_bbox);
    }

    static void findActivePixels(const cv::Mat& mat, std::vector<CvPoint>& points)
    {
        CV_Assert(mat.type() == CV_8UC3);

        for (int r = 0; r < mat.rows; r++) {
            for (int c = 0; c < mat.cols; c++) {
                const uchar* ren_data = mat.at<unsigned char[3]>(r, c);
                unsigned char cb = *ren_data++;
                unsigned char cg = *ren_data++;
                unsigned char cr = *ren_data++;
                if (!(cb == 0 && cg == 0 && cr == 0)) {
                    points.push_back(cv::Point(r, c));
                    //std::cout << "Target 1: row: " << r << " col: " << c << std::endl;
                }
            }
        }
    }

    void FaceSwap::paintMappings()
    {
        std::cout << "Mapping paintings:"
            << std::endl
            << "========================="
            << std::endl;
        for (const auto& pointsPair : m_pixelMappings)
        {
            std::cout << "Source: (" << pointsPair.first.x
                << "," << pointsPair.first.y << "): ";

            std::stringstream ss2;
            ss2 << "C:\\face_swap\\pics\\brad_"
                << pointsPair.first.x  << "_"
                << pointsPair.first.y << ".jpg";
            std::string img_path2 = ss2.str();
            //cv::Mat img = cv::imread("C:\\face_swap\\pics\\brad.jpg");
            cv::Mat img = cv::imread("C:\\face_swap\\pics\\img_cropped.jpg");
            cv::imwrite(img_path2, img);

            draw_green_pixel(
                img_path2,
                img_path2,
                pointsPair.first.x,
                pointsPair.first.y);


            std::stringstream ss;
            ss << "C:\\face_swap\\pics\\rendered_"
               << pointsPair.first.x  << "_"
               << pointsPair.first.y << ".jpg";

            std::string img_path = ss.str();

            img = cv::imread("C:\\face_swap\\pics\\rendered.jpg");
            cv::imwrite(img_path, img);

            for (const auto& targetPoints : pointsPair.second)
            {
                int x = targetPoints.x, y = targetPoints.y;
                draw_green_pixel(
                    img_path,
                    img_path,
                    x,
                    y);
            }

            std::cout << std::endl;
        }
    }

    void FaceSwap::dumpMappings()
    {
        const char* path = "mappings.txt";
        std::ofstream file(path, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("unable to open file: ") + path);
        }
        std::cout << "Dumping mappings to: " << path << std::endl;
        file << std::string("Source Point, Target Points") << std::endl;

        int i = 0;
        int size = m_absolutePixelMappings.size();
        bool draw_on_target = false;
        //for (const auto& pointsPair : m_absolutePixelMappings)
        //{
        //    if (!draw_on_target && i == size / 2) {
        //        draw_green_pixel(
        //            "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\001.png",
        //            "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\green_pixel_src.png",
        //            pointsPair.first.x,
        //            pointsPair.first.y);
        //        draw_on_target = true;
        //    }
        //    file << pointsPair.first.x << "," << pointsPair.first.y << " ";
        //    for (const auto& targetPoints : pointsPair.second)
        //    {
        //        if (draw_on_target) {
        //            draw_green_pixel(
        //                "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\002.png",
        //                "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\green_pixel_dst.png",
        //                targetPoints.x,
        //                targetPoints.y);
        //            draw_on_target = false;
        //        }
        //        file << targetPoints.x << ","
        //            << targetPoints.y << " ";
        //    }
        //    file << std::endl;
        //    i++;
        //}
        //std::cout << "Mappings of points:"
        //          << std::endl
        //          << "========================="
        //          << std::endl;
        for (const auto& pointsPair : m_pixelMappings)
        {
            if (!draw_on_target && i == size / 2) {
                draw_green_pixel(
                    "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\cropped_src.png",
                    "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\green_pixel_cropped_src.png",
                    pointsPair.first.x,
                    pointsPair.first.y);
                draw_on_target = true;
            }


            file << pointsPair.first.x << "," << pointsPair.first.y << " ";
            //std::cout << "Source: (" << pointsPair.first.x
            //          << "," << pointsPair.first.y << "): ";
            for (const auto& targetPoints : pointsPair.second)
            {
                file << targetPoints.x << ","
                     << targetPoints.y << " ";
                if (draw_on_target) {
                    draw_green_pixel(
                        "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\cropped_tgt.png",
                        "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\green_pixel_cropped_tgt.png",
                        targetPoints.x,
                        targetPoints.y);
                    draw_on_target = false;
                }
                //std::cout << "(" << targetPoints.x
                //          << "," << targetPoints.y << "), ";
            }
            file << std::endl;
            i++;
            //std::cout << std::endl;
        }
    }

    void FaceSwap::dumpAbsoluteMappings()
    {
        const char* path = "absolute_mappings.txt";
        std::ofstream file(path, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("unable to open file: ") + path);
        }

        std::cout << "Dumping absolute mappings to: " << path << std::endl;
        file << std::string("Source Point, Target Points") << std::endl;

        int i = 0;
        int size = m_absolutePixelMappings.size();
        bool draw_on_target = false;
        for (const auto& pointsPair : m_absolutePixelMappings)
        {
            if (!draw_on_target && i == size / 2) {
                draw_green_pixel(
                    "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\001.png",
                    "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\green_pixel_src.png",
                    pointsPair.first.x,
                    pointsPair.first.y);
                draw_on_target = true;
            }
            file << pointsPair.first.x << "," << pointsPair.first.y << " ";
            for (const auto& targetPoints : pointsPair.second)
            {
                if (draw_on_target) {
                    draw_green_pixel(
                        "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\002.png",
                        "C:\\face_swap\\face_swap\\build\\install\\data\\images\\bbai5a\\green_pixel_dst.png",
                        targetPoints.x,
                        targetPoints.y);
                    draw_on_target = false;
                }
                file << targetPoints.x << ","
                     << targetPoints.y << " ";
            }
            file << std::endl;
            i++;
        }
    }

    void FaceSwap::dumpStats()
    {
        const char* path = "statistics.txt";
        std::ofstream file(path, std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("unable to open file: ") + path);
        }
        size_t numPixels = m_tex_num_white_pixels;
        float mappedPercentage = (float)m_pixelMappings.size() / numPixels;
        std::cout << "Dumping statistics to: " << path << std::endl;
        file << "Total number of pixels:" << numPixels << std::endl;
        file << "Total number of mapped pixels:" << m_pixelMappings.size() << std::endl;
        file << "Percentage of mapped pixels: " << mappedPercentage * 100 << "%" << std::endl;
        file << "Source bounding box "
             << "x = " << m_source_bbox.x << ", "
             << "y = " << m_source_bbox.y << ", "
             << "width = " << m_source_bbox.width << ", "
             << "height = " << m_source_bbox.height << std::endl;
        file << "Target bounding box "
             << "x = " << m_target_bbox.x << ", "
             << "y = " << m_target_bbox.y << ", "
             << "width = " << m_target_bbox.width << ", "
             << "height = " << m_target_bbox.height << std::endl;
        file << "m_score : "
            << "0 : " << m_score.val[0] << ", "
            << "1 : " << m_score.val[1] << ", "
            << "2 : " << m_score.val[2] << std::endl;
        file << "Score Avg : "
            << ((m_score.val[0] + m_score.val[1] + m_score.val[2]) / 3)
            << std::endl;
    }

    void FaceSwap::generateAbsoluteMappings()
    {
        CV_Assert(m_pixelMappings.size() > 0 && "No mappings were found! "
                                                "Did you call generateMappings()?");

        for (const auto& pointsPair : m_pixelMappings)
        {
            // TODO: understand why we add y and not x
            int absSrcX = pointsPair.first.x + m_source_bbox.y;
            int absSrcY = pointsPair.first.y + m_source_bbox.x;

            std::vector<CvPoint> points;

            for (const auto& targetPoints : pointsPair.second)
            {
                int absDstX = targetPoints.x + m_target_bbox.y;
                int absDstY = targetPoints.y + m_target_bbox.x;
                points.push_back(CvPoint(absDstX, absDstY));
            }

            m_absolutePixelMappings.push_back(std::make_pair(CvPoint(absSrcX, absSrcY), points));
        }

    }

    void FaceSwap::generateMappings()
    {
        CV_Assert(m_tex.type() == CV_8UC4);

        std::vector<cv::Mat> channels;
        cv::split(m_tex, channels);
        CV_Assert(channels.size() == 4);

        cv::Mat tex_seg = channels.back().clone();
        cv::Mat cur_tex;

        m_tex_num_white_pixels = 0;

        // 1. Iterate over all face pixels in the source images
        CV_Assert(tex_seg.type() == CV_8UC1);
        for (int r = 0; r < m_tex.rows; r++)
        {
            for (int c = 0; c < m_tex.cols; c++)
            {
                // Nose: Source: row: 210, col : 165
                // Right Eyebrows: row: 163 col : 233
                // Left Eyebrows: row: 163 col: 100
                //if (!(r == 283 && c == 209 || 
                //      r == 256 && c == 256 ||
                //      r == 350 && c == 256))
                //    continue;

                if (tex_seg.at<uchar>(r, c) == 255)
                {
                    m_tex_num_white_pixels++;

                    // 2. Set a pixel as the only active pixel
                    cur_tex = cv::Mat::zeros(m_tex.size(), m_tex.type());
                    unsigned char* data = cur_tex.at<unsigned char[4]>(r, c);
                    *data++ = 255;
                    *data++ = 255;
                    *data++ = 255;
                    *data++ = 255;

                    // Prepare mesh
                    m_dst_mesh.tex = cur_tex;
                    m_face_renderer->setMesh(m_dst_mesh);

                    // Render
                    cv::Mat rendered_img;
                    m_face_renderer->render(m_vecR, m_vecT);
                    m_face_renderer->getFrameBuffer(rendered_img);

                    std::vector<CvPoint> points;
                    findActivePixels(rendered_img, points);

                    // 3. Retrieve corresponding pixels in the target image
                    if (points.size() > 0) {
                        m_pixelMappings.push_back(std::make_pair(CvPoint(r, c), points));
                    }
                }
            }
        }
    }

    void FaceSwap::generateSourceMappings()
    {
        // Source texture - 255 == face pixel, 0 == background.
        cv::Mat src_tex = cv::Mat::zeros(m_source_img.size(), CV_8UC1);

        CV_Assert(m_tex.type() == CV_8UC4);

        std::vector<cv::Mat> channels;
        cv::split(m_tex, channels);
        CV_Assert(channels.size() == 4);

        cv::Mat tex_seg = channels.back().clone();
        cv::Mat cur_tex;

        // 1. Iterate over all face pixels in the source images
        CV_Assert(tex_seg.type() == CV_8UC1);
        for (int r = 0; r < m_tex.rows; r++)
        {
            for (int c = 0; c < m_tex.cols; c++)
            {
                if (tex_seg.at<uchar>(r, c) == 255)
                {
                    // 2. Get absolute pixel location.
                    // TODO: understand why we add y and not x
                    int absSrcX = r + m_source_bbox.y;
                    int absSrcY = c + m_source_bbox.x;

                    src_tex.at<uchar>(absSrcX, absSrcY) = 255;
                }
            }
        }

        // 3. Dump image
        cv::imwrite("c:\\face_swap\\temp\\out.jpg", src_tex);
    }

    cv::Mat FaceSwap::swap()
    {
        // Render
        cv::Mat rendered_img;
        m_face_renderer->render(m_vecR, m_vecT);
        m_face_renderer->getFrameBuffer(rendered_img);

        // 1) Subtract W(I1) - I2
        // min value is -255 0   - 255
        // max value is +255 255 - 0
        // TODO: need to do range mapping: ((value - min) / (max - min)) * 255
        cv::Mat diff_image_rgb = rendered_img - m_W_i1;

        //cv::normalize(0, 255, diff_image_rgb, MINMAX)

        // 2) Convert (RGB) diff image to grayscale
        //cv::Mat diff_image_grayscale;
        //cv::cvtColor(diff_image_rgb, diff_image_grayscale, cv::COLOR_RGB2GRAY);

        /*uchar* ren_data = rendered_img.data;*/
        //CV_Assert(rendered_img.type() == CV_8UC3);
        //for (int r = 0; r < rendered_img.rows; r++) {
        //    for (int c = 0; c < rendered_img.cols; c++) {
        //        uchar* ren_data = rendered_img.at<unsigned char[3]>(r, c);
        //        unsigned char cb = *ren_data++;
        //        unsigned char cg = *ren_data++;
        //        unsigned char cr = *ren_data++;
        //        if (!(cb == 0 && cg == 0 && cr == 0)) {
        //            std::cout << "Target 1: row: " << r << " col: " << c << std::endl;
        //        }
        //    }
        //}

        std::cout << "rendered: " << rendered_img.size() << std::endl;

        boost::filesystem::path out_file_path = "c:\\face_swap\\rendered.jpg";
        cv::imwrite(out_file_path.string(), rendered_img);

        //draw_green_pixel(
        //    "C:\\face_swap\\pics\\rendered.jpg",
        //    "C:\\face_swap\\pics\\rendered_green.jpg",
        //    155,
        //    156);
        
        // Blend images
        cv::Mat tgt_rendered_img = cv::Mat::zeros(m_target_img.size(), CV_8UC3);
        rendered_img.copyTo(tgt_rendered_img(m_target_bbox));

        m_tgt_rendered_img = tgt_rendered_img;  // For debug

        /*uchar* ren_data = rendered_img.data;*/
        //CV_Assert(tgt_rendered_img.type() == CV_8UC3);
        //for (int r = 0; r < tgt_rendered_img.rows; r++) {
        //    for (int c = 0; c < tgt_rendered_img.cols; c++) {
        //        uchar* ren_data = tgt_rendered_img.at<unsigned char[3]>(r, c);
        //        unsigned char cb = *ren_data++;
        //        unsigned char cg = *ren_data++;
        //        unsigned char cr = *ren_data++;
        //        if (!(cb == 0 && cg == 0 && cr == 0)) {
        //            std::cout << "Target 2: row: " << r << " col: " << c << std::endl;
        //        }
        //    }
        //}

        //draw_green_pixel(
        //    "C:\\face_swap\\pics\\tgt_rendered_img.jpg",
        //    "C:\\face_swap\\pics\\tgt_rendered_img_green.jpg",
        //    155,
        //    167);

        //out_file_path = "c:\\face_swap\\tgt_rendered_img.jpg";
        //cv::imwrite(out_file_path.string(), tgt_rendered_img);

        return blend(tgt_rendered_img, m_target_img, m_target_seg);
    }

    const Mesh & FaceSwap::getSourceMesh() const
    {
        return m_src_mesh;
    }

    const Mesh & FaceSwap::getTargetMesh() const
    {
        return m_dst_mesh;
    }

    bool FaceSwap::preprocessImages(const cv::Mat& img, const cv::Mat& seg,
        std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
        cv::Mat& cropped_img, cv::Mat& cropped_seg, cv::Rect& bbox)
    {
        // Calculate landmarks
        m_sfl->clear();
        const sfl::Frame& lmsFrame = m_sfl->addFrame(img);
        if (lmsFrame.faces.empty()) return false;
        //std::cout << "faces found = " << lmsFrame.faces.size() << std::endl;    // Debug
        const sfl::Face* face = lmsFrame.getFace(sfl::getMainFaceID(m_sfl->getSequence()));
        landmarks = face->landmarks; // Debug
        cropped_landmarks = landmarks; 

        // Calculate crop bounding box
        bbox = sfl::getFaceBBoxFromLandmarks(landmarks, img.size(), true);
        bbox.width = bbox.width / 4 * 4;    // Make sure cropped image is dividable by 4
        bbox.height = bbox.height / 4 * 4;

        // Crop landmarks
        for (cv::Point& p : cropped_landmarks)
        {
            p.x -= bbox.x;
            p.y -= bbox.y;
        }

        // Crop images
        cropped_img = img(bbox);
        if(!seg.empty()) cropped_seg = seg(bbox);

        return true;
    }

    bool FaceSwap::preprocessImages(const cv::Mat& img, const cv::Mat& seg,
        std::vector<cv::Point>& landmarks, std::vector<cv::Point>& cropped_landmarks,
        cv::Mat& cropped_img, cv::Mat& cropped_seg)
    {
        cv::Rect bbox;
        return preprocessImages(img, seg, landmarks, cropped_landmarks,
            cropped_img, cropped_seg, bbox);
    }

    void FaceSwap::generateTexture(const Mesh& mesh, const cv::Mat& img, 
        const cv::Mat& seg, const cv::Mat& vecR, const cv::Mat& vecT,
        const cv::Mat& K, cv::Mat& tex, cv::Mat& uv)
    {
        // Resize images to power of 2 size
        cv::Size tex_size(nextPow2(img.cols), nextPow2(img.rows));
        cv::Mat img_scaled, seg_scaled;
        cv::resize(img, img_scaled, tex_size, 0.0, 0.0, cv::INTER_CUBIC);
        if(!seg.empty())
            cv::resize(seg, seg_scaled, tex_size, 0.0, 0.0, cv::INTER_NEAREST);

        // Combine image and segmentation into one 4 channel texture
        if (!seg.empty())
        {
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            channels.push_back(seg);
            cv::merge(channels, tex);
            
            boost::filesystem::path out_tex = "c:\\face_swap\\img.jpg";
            cv::imwrite(out_tex.string(), img);
            out_tex = "c:\\face_swap\\seg.jpg";
            cv::imwrite(out_tex.string(), seg);
            out_tex = "c:\\face_swap\\tex_merged.jpg";
            cv::imwrite(out_tex.string(), tex);

        }
        else tex = img_scaled; 

        uv = generateTextureCoordinates(m_src_mesh, img.size(), vecR, vecT, K);
    }

    cv::Mat FaceSwap::generateTextureCoordinates(
        const Mesh& mesh,const cv::Size& img_size,
        const cv::Mat & vecR, const cv::Mat & vecT, const cv::Mat & K)
    {
        cv::Mat P = createPerspectiveProj3x4(vecR, vecT, K);
        cv::Mat pts_3d;
        cv::vconcat(mesh.vertices.t(), cv::Mat::ones(1, mesh.vertices.rows, CV_32F), pts_3d);
        cv::Mat proj = P * pts_3d;

        // Normalize projected points
        cv::Mat uv(mesh.vertices.rows, 2, CV_32F);
        float* uv_data = (float*)uv.data;
        float z;
        for (int i = 0; i < uv.rows; ++i)
        {
            z = proj.at<float>(2, i);
            *uv_data++ = proj.at<float>(0, i) / (z * img_size.width);
            *uv_data++ = proj.at<float>(1, i) / (z * img_size.height);
        }

        return uv;
    }

    cv::Mat FaceSwap::blend(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& dst_seg)
    {
        // Calculate mask
        cv::Mat mask(src.size(), CV_8U);
        unsigned char* src_data = src.data;
        unsigned char* dst_seg_data = dst_seg.data;
        unsigned char* mask_data = mask.data;
        for (int i = 0; i < src.total(); ++i)
        {
            unsigned char cb = *src_data++;
            unsigned char cg = *src_data++;
            unsigned char cr = *src_data++;
            if (!(cb == 0 && cg == 0 && cr == 0))  *mask_data++ = 255;
            else *mask_data++ = 0;
        }

        // Combine the segmentation with the mask
        if (!dst_seg.empty())
            cv::bitwise_and(mask, dst_seg, mask);

        // Find center point
        int minc = std::numeric_limits<int>::max(), minr = std::numeric_limits<int>::max();
        int maxc = 0, maxr = 0;
        mask_data = mask.data;
        for (int r = 0; r < src.rows; ++r)
            for (int c = 0; c < src.cols; ++c)
            {
                if (*mask_data++ < 255) continue;
                minc = std::min(c, minc);
                minr = std::min(r, minr);
                maxc = std::max(c, maxc);
                maxr = std::max(r, maxr);
            }
        if (minc >= maxc || minr >= maxr) return cv::Mat();
        cv::Point center((minc + maxc) / 2, (minr + maxr) / 2);

        /// Debug ///
        //cv::Mat out = src.clone();
        //cv::rectangle(out, cv::Point(minc, minr), cv::Point(maxc, maxr), cv::Scalar(255, 0, 0));
        //cv::imshow("target", out);
        //cv::imshow("mask", mask);
        //cv::waitKey(0);
        /////////////

        // Do blending
        cv::Mat blend;
        cv::seamlessClone(src, dst, mask, center, blend, cv::NORMAL_CLONE);

        return blend;
    }

    cv::Mat FaceSwap::debugSourceMeshWireframe()
    {
        const float scale = 3.0f;
        cv::Mat out = m_src_cropped_img.clone();
        cv::Mat P = createPerspectiveProj3x4(m_src_vecR, m_src_vecT, m_src_K);
        std::vector<cv::Point> scaled_landmarks(m_src_cropped_landmarks);
        for (cv::Point& p : scaled_landmarks) p *= (int)scale;

        // Render
        renderWireframe(out, m_src_mesh, P, scale);
        //sfl::render(out, scaled_landmarks, false, cv::Scalar(0, 0, 255));
        return out;
    }

    cv::Mat FaceSwap::debugTargetMeshWireframe()
    {
        const float scale = 3.0f;
        cv::Mat out = m_tgt_cropped_img.clone();
        cv::Mat P = createPerspectiveProj3x4(m_vecR, m_vecT, m_K);
        std::vector<cv::Point> scaled_landmarks(m_tgt_cropped_landmarks);
        for (cv::Point& p : scaled_landmarks) p *= (int)scale;

        // Render
        renderWireframe(out, m_dst_mesh, P, scale);
        //sfl::render(out, scaled_landmarks, false, cv::Scalar(0, 0, 255));
        return out;
    }

    cv::Mat FaceSwap::debug()
    {
        cv::Mat src_d = debugSourceMeshWireframe();
        cv::Mat tgt_d = debugTargetMeshWireframe();
        cv::Size max_size(std::max(src_d.cols, tgt_d.cols), 
            std::max(src_d.rows, tgt_d.rows));

        cv::Mat src_d_out = cv::Mat::zeros(max_size, CV_8UC3);
        cv::Mat tgt_d_out = cv::Mat::zeros(max_size, CV_8UC3);
        src_d.copyTo(src_d_out(cv::Rect(0, 0, src_d.cols, src_d.rows)));
        tgt_d.copyTo(tgt_d_out(cv::Rect(0, 0, tgt_d.cols, tgt_d.rows)));
        cv::Mat out;
        cv::hconcat(src_d_out, tgt_d_out, out);
        return out;
    }

    cv::Mat FaceSwap::debugSourceMesh()
    {
        return debugMesh(m_src_cropped_img, m_src_cropped_seg, m_uv, 
            m_src_mesh, m_src_vecR, m_src_vecT, m_src_K);
    }

    cv::Mat FaceSwap::debugTargetMesh()
    {
        cv::Mat uv = generateTextureCoordinates(m_dst_mesh, m_tgt_cropped_seg.size(),
            m_vecR, m_vecT, m_K);
        return debugMesh(m_tgt_cropped_img, m_tgt_cropped_seg, uv,
            m_dst_mesh, m_vecR, m_vecT, m_K);
    }

    cv::Mat FaceSwap::debugMesh(const cv::Mat& img, const cv::Mat& seg,
        const cv::Mat& uv, const Mesh& mesh,
        const cv::Mat& vecR, const cv::Mat& vecT, const cv::Mat& K)
    {
        cv::Mat out;

        // Create texture
        cv::Mat tex(img.size(), CV_8UC3);
        unsigned char* tex_data = tex.data;
        int total_pixels = tex.total() * tex.channels();
        for (int i = 0; i < total_pixels; ++i) *tex_data++ = 192;

        // Add segmentation colors
        if (!seg.empty())
        {
            cv::Vec3b* tex_data = (cv::Vec3b*)tex.data;
            unsigned char* seg_data = seg.data;
            for (int i = 0; i < tex.total(); ++i)
            {
                //if (*seg_data++ > 0)
                if (seg.at<unsigned char>(i) > 0)
                {
                    (*tex_data)[0] = 0;
                    (*tex_data)[1] = 0;
                    (*tex_data)[2] = 240;
                }
                ++tex_data;
            }
        }

        cv::Size tex_size(nextPow2(img.cols), nextPow2(img.rows));
        cv::resize(tex, tex, tex_size, 0.0, 0.0, cv::INTER_CUBIC);

        // Initialize mesh
        Mesh tmp_mesh = mesh;
        tmp_mesh.tex = tex;
        tmp_mesh.uv = uv;
        tmp_mesh.normals = computeVertexNormals(tmp_mesh);

        // Render
        m_face_renderer->init(img.cols, img.rows);
        m_face_renderer->setProjection(K.at<float>(4));
        m_face_renderer->setMesh(tmp_mesh);

        cv::Mat pos_dir = (cv::Mat_<float>(4, 1) << -0.25f, -0.5f, -1, 0);
        cv::Mat ambient = (cv::Mat_<float>(4, 1) << 0.3f, 0.3f, 0.3f, 1);
        cv::Mat diffuse = (cv::Mat_<float>(4, 1) << 1.0f, 1.0f, 1.0f, 1);
        m_face_renderer->setLight(pos_dir, ambient, diffuse);
        m_face_renderer->render(vecR, vecT);
        m_face_renderer->clearLight();

        m_face_renderer->getFrameBuffer(out);

        // Overwrite black pixels with original pixels
        cv::Vec3b* out_data = (cv::Vec3b*)out.data;
        for (int i = 0; i < out.total(); ++i)
        {
            unsigned char b = (*out_data)[0];
            unsigned char g = (*out_data)[1];
            unsigned char r = (*out_data)[2];
            if (b == 0 && g == 0 && r == 0)
                *out_data = img.at<cv::Vec3b>(i);
            ++out_data;
        }

        return out;
    }

    cv::Mat FaceSwap::debugSourceLandmarks()
    {
        cv::Mat out = m_source_img.clone();
        sfl::render(out, m_src_landmarks);
        return out;
    }

    cv::Mat FaceSwap::debugTargetLandmarks()
    {
        cv::Mat out = m_target_img.clone();
        sfl::render(out, m_tgt_landmarks);
        return out;
    }

    cv::Mat FaceSwap::debugRender()
    {
        cv::Mat out = m_tgt_rendered_img.clone();

        // Overwrite black pixels with original pixels
        cv::Vec3b* out_data = (cv::Vec3b*)out.data;
        for (int i = 0; i < out.total(); ++i)
        {
            unsigned char b = (*out_data)[0];
            unsigned char g = (*out_data)[1];
            unsigned char r = (*out_data)[2];
            if (b == 0 && g == 0 && r == 0)
                *out_data = m_target_img.at<cv::Vec3b>(i);
            ++out_data;
        }

        return out;
    }

}   // namespace face_swap