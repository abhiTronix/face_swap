// std
#include <iostream>
#include <exception>

// Boost
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// sfl
#include <sfl/sequence_face_landmarks.h>
#include <sfl/utilities.h>

// face_swap
#include <face_swap/face_swap.h>

// OpenGL
#include <GL/glew.h>

// Qt
#include <QApplication>
#include <QOpenGLContext>
#include <QOffscreenSurface>

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <cassert>

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::runtime_error;
using namespace boost::program_options;
using namespace boost::filesystem;

#if 1
#undef NDEBUG

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}
typedef std::pair<CvPoint, std::vector<CvPoint>> PixelsPair;
std::vector<PixelsPair> g_absolutePixelMappings;

void get_w(const char* w_path)
{
    std::ifstream file(w_path);
    if (!file.is_open())
        throw std::runtime_error("unable to open file.");

    std::string line;
    int i = 0;
    for ( ; std::getline(file, line, '\n'); i++) {
        if (i == 0) {
            continue;
        }

        std::vector<std::string> splt = split(line, ' ');

        bool is_source_point = true;
        unsigned src_x, src_y;
        std::vector<CvPoint> points;
        for (const auto& word : splt) {
            unsigned x, y;
            //std::cout << word << " ";
            sscanf(word.c_str(), "%u,%u", &x, &y);
            if (is_source_point) {
                src_x = x;
                src_y = y;
                is_source_point = false;
            }
            else {
                points.push_back(CvPoint(x, y));
            }

        }

        g_absolutePixelMappings.push_back(std::make_pair(CvPoint(src_x, src_y), points));
    }

}

void apply_w_impl(const cv::Mat& j1, const cv::Mat& j2) //, const cv::Rect& bbox)
{
    CV_Assert(j1.type() == CV_8UC3);
    cv::Mat W_j1 = cv::Mat::zeros(j1.size(), j1.type());
    cv::Mat W_mask = cv::Mat::zeros(j1.size(), j1.type());
    const uchar mask_vals[] = { 1, 1, 1 };
    for (const auto& pointsPair : g_absolutePixelMappings)
    {
        const uchar* src_pixel = j1.at<unsigned char[3]>(pointsPair.first.x, pointsPair.first.y);
        for (const auto& targetPoints : pointsPair.second)
        {
            uchar* dst_pixel = W_j1.at<unsigned char[3]>(targetPoints.x, targetPoints.y);
            uchar* dst_mask_pixel = W_mask.at<unsigned char[3]>(targetPoints.x, targetPoints.y);
            // TODO: there must be a better way of copying pixels!
            memcpy(dst_pixel, src_pixel, sizeof(unsigned char[3]));

            // Build target mask
            memcpy(dst_mask_pixel, mask_vals, sizeof(mask_vals));

            //pow(diff_image_rgb .* dst_mask_pixel);
        }
    }

    cv::Mat pow_img_rgb;
    cv::Mat diff_img_rgb_float;
    cv::Mat diff_img_rgb = cv::max(j2, W_j1) - cv::min(j2, W_j1);
    diff_img_rgb = diff_img_rgb.mul(W_mask);
    diff_img_rgb.convertTo(diff_img_rgb_float, CV_32F);
    cv::pow(diff_img_rgb_float, 2, pow_img_rgb);
    cv::Scalar score = cv::mean(pow_img_rgb);

    std::cout << "score : "
              << "0 : " << score.val[0] << ", "
              << "1 : " << score.val[1] << ", "
              << "2 : " << score.val[2] << std::endl;

    std::cout << "Score Avg : "
              << ((score.val[0] + score.val[1] + score.val[2]) / 3)
              << std::endl;

    std::string out_path = "W_j1_new.png";
    cv::imwrite(out_path, W_j1);
}

void apply_w(const cv::Mat& j1, const cv::Mat& j2)
{
    const char* file_path = getenv("FACE_SWAP_W_PATH");
    const char* bbox_value = getenv("FACE_SWAP_BBOX");

    if (!file_path) {
        throw std::invalid_argument("FACE_SWAP_W_PATH is not set.");
    }

    if (!bbox_value) {
        throw std::invalid_argument("FACE_SWAP_BBOX is not set.");
    }

    //unsigned bbox_x, bbox_y, width, height;
    //sscanf(bbox_value, "%u,%u,%u,%u", &bbox_x, &bbox_y, &width, &height);
    //cv::Rect bbox(bbox_x, bbox_y, width, height);

    get_w(file_path);
    if (g_absolutePixelMappings.empty()) {
        throw std::invalid_argument("no entries found.");
    }
    assert(absolutePixelMappings.size() > 0);
    apply_w_impl(j1, j2); //, bbox);
    //exit(1);
}

#define NDEBUG
#endif

int main(int argc, char* argv[])
{
	// Parse command line arguments
    std::vector<string> input_paths, seg_paths;
	string output_path, landmarks_path;
	string model_3dmm_h5_path, model_3dmm_dat_path;
	string reg_model_path, reg_deploy_path, reg_mean_path;
	string seg_model_path, seg_deploy_path;
    string cfg_path;
    bool generic, with_expr, with_gpu;
    unsigned int gpu_device_id, verbose;
	try {
		options_description desc("Allowed options");
		desc.add_options()
			("help,h", "display the help message")
            ("verbose,v", value<unsigned int>(&verbose)->default_value(0), "output debug information [0, 4]")
			("input,i", value<std::vector<string>>(&input_paths)->required(), "image paths [source target]")
			("output,o", value<string>(&output_path)->required(), "output path")
            ("segmentations,s", value<std::vector<string>>(&seg_paths), "segmentation paths [source target]")
			("landmarks,l", value<string>(&landmarks_path)->required(), "path to landmarks model file")
            ("model_3dmm_h5", value<string>(&model_3dmm_h5_path)->required(), "path to 3DMM file (.h5)")
            ("model_3dmm_dat", value<string>(&model_3dmm_dat_path)->required(), "path to 3DMM file (.dat)")
            ("reg_model,r", value<string>(&reg_model_path)->required(), "path to 3DMM regression CNN model file (.caffemodel)")
            ("reg_deploy,d", value<string>(&reg_deploy_path)->required(), "path to 3DMM regression CNN deploy file (.prototxt)")
            ("reg_mean,m", value<string>(&reg_mean_path)->required(), "path to 3DMM regression CNN mean file (.binaryproto)")
			("seg_model", value<string>(&seg_model_path), "path to face segmentation CNN model file (.caffemodel)")
			("seg_deploy", value<string>(&seg_deploy_path), "path to face segmentation CNN deploy file (.prototxt)")
            ("generic,g", value<bool>(&generic)->default_value(false), "use generic model without shape regression")
            ("expressions,e", value<bool>(&with_expr)->default_value(true), "with expressions")
			("gpu", value<bool>(&with_gpu)->default_value(true), "toggle GPU / CPU")
			("gpu_id", value<unsigned int>(&gpu_device_id)->default_value(0), "GPU's device id")
            ("cfg", value<string>(&cfg_path)->default_value("face_swap_image.cfg"), "configuration file (.cfg)")
			;
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).
			positional(positional_options_description().add("input", -1)).run(), vm);

        if (vm.count("help")) {
            cout << "Usage: face_swap_image [options]" << endl;
            cout << desc << endl;
            exit(0);
        }

        // Read config file
        std::ifstream ifs(vm["cfg"].as<string>());
        store(parse_config_file(ifs, desc), vm);

        notify(vm);

        if(input_paths.size() != 2) throw error("Both source and target must be specified in input!");
        if (!is_regular_file(input_paths[0])) throw error("source input must be a path to an image!");
        if (!is_regular_file(input_paths[1])) throw error("target input target must be a path to an image!");
        if (seg_paths.size() > 0 && !is_regular_file(seg_paths[0]))
            throw error("source segmentation must be a path to an image!");
        if (seg_paths.size() > 1 && !is_regular_file(seg_paths[1]))
            throw error("target segmentation must be a path to an image!");
		if (!is_regular_file(landmarks_path)) throw error("landmarks must be a path to a file!");
        if (!is_regular_file(model_3dmm_h5_path)) throw error("model_3dmm_h5 must be a path to a file!");
        if (!is_regular_file(model_3dmm_dat_path)) throw error("model_3dmm_dat must be a path to a file!");
        if (!is_regular_file(reg_model_path)) throw error("reg_model must be a path to a file!");
        if (!is_regular_file(reg_deploy_path)) throw error("reg_deploy must be a path to a file!");
        if (!is_regular_file(reg_mean_path)) throw error("reg_mean must be a path to a file!");
		if (!seg_model_path.empty() && !is_regular_file(seg_model_path))
			throw error("seg_model must be a path to a file!");
		if (!seg_deploy_path.empty() && !is_regular_file(seg_deploy_path))
			throw error("seg_deploy must be a path to a file!");
	}
	catch (const error& e) {
        cerr << "Error while parsing command-line arguments: " << e.what() << endl;
        cerr << "Use --help to display a list of options." << endl;
		exit(1);
	}

	try
	{
        // Intialize OpenGL context
        QApplication a(argc, argv);

        QSurfaceFormat surfaceFormat;
        surfaceFormat.setMajorVersion(1);
        surfaceFormat.setMinorVersion(5);

        QOpenGLContext openGLContext;
        openGLContext.setFormat(surfaceFormat);
        openGLContext.create();
        if (!openGLContext.isValid()) return -1;

        QOffscreenSurface surface;
        surface.setFormat(surfaceFormat);
        surface.create();
        if (!surface.isValid()) return -2;

        openGLContext.makeCurrent(&surface);

        // Initialize GLEW
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            // Problem: glewInit failed, something is seriously wrong
            fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
            throw std::runtime_error("Failed to initialize GLEW!");
        }

        // Initialize face swap
        face_swap::FaceSwap fs(landmarks_path, model_3dmm_h5_path, model_3dmm_dat_path,
            reg_model_path, reg_deploy_path, reg_mean_path, generic, with_expr,
			with_gpu, (int)gpu_device_id);

        // Read source and target images
        cv::Mat source_img = cv::imread(input_paths[0]);
        cv::Mat target_img = cv::imread(input_paths[1]);

        //apply_w(source_img, target_img);

        // Read source and target segmentations or initialize segmentation model
        cv::Mat source_seg, target_seg;
		if (seg_model_path.empty() || seg_deploy_path.empty())
		{
			if (seg_paths.size() > 0) 
				source_seg = cv::imread(seg_paths[0], cv::IMREAD_GRAYSCALE);
			if (seg_paths.size() > 1) 
				target_seg = cv::imread(seg_paths[1], cv::IMREAD_GRAYSCALE);
		}
		else fs.setSegmentationModel(seg_model_path, seg_deploy_path);

        // Set source and target
		/*
        if (!fs.setSource(source_img, source_seg))
            throw std::runtime_error("Failed to find faces in source image!");
        if (!fs.setTarget(target_img, target_seg))
            throw std::runtime_error("Failed to find faces in target image!");
			*/
		if (!fs.setImages(source_img, target_img, source_seg, target_seg))
			throw std::runtime_error("Failed to find faces in one of the images!");

        // Do face swap
        cv::Mat rendered_img = fs.swap();
        if (rendered_img.empty())
            throw std::runtime_error("Face swap failed!");

        // Write output to file
        path out_file_path = output_path;
		path out_dir_path = output_path;
        if (is_directory(output_path))
        {
            path outputName = (path(input_paths[0]).stem() += "_") += 
                (path(input_paths[1]).stem() += ".jpg");
			out_file_path = path(output_path) /= outputName;
        }
		else out_dir_path = path(output_path).parent_path();
        cv::imwrite(out_file_path.string(), rendered_img);

        // Debug
        if (verbose > 0)
        {
			// Write rendered image
			path debug_render_path = out_dir_path /
				(out_file_path.stem() += "_render.jpg");
			cv::Mat debug_render_img = fs.debugRender();
			cv::imwrite(debug_render_path.string(), debug_render_img); 
        }
		if (verbose > 1)
		{
			// Write projected meshes
			path debug_src_mesh_path = out_dir_path /
				(out_file_path.stem() += "_src_mesh.jpg");
			cv::Mat debug_src_mesh_img = fs.debugSourceMesh();
			cv::imwrite(debug_src_mesh_path.string(), debug_src_mesh_img);

			path debug_tgt_mesh_path = out_dir_path /
				(out_file_path.stem() += "_tgt_mesh.jpg");
			cv::Mat debug_tgt_mesh_img = fs.debugTargetMesh();
			cv::imwrite(debug_tgt_mesh_path.string(), debug_tgt_mesh_img);
		}
		if (verbose > 2)
		{
			// Write landmarks render
			path debug_src_lms_path = out_dir_path /
				(out_file_path.stem() += "_src_landmarks.jpg");
			cv::Mat debug_src_lms_img = fs.debugSourceLandmarks();
			cv::imwrite(debug_src_lms_path.string(), debug_src_lms_img);

			path debug_tgt_lms_path = out_dir_path /
				(out_file_path.stem() += "_tgt_landmarks.jpg");
			cv::Mat debug_tgt_lms_img = fs.debugTargetLandmarks();
			cv::imwrite(debug_tgt_lms_path.string(), debug_tgt_lms_img);
		}
		if (verbose > 3)
		{
			// Write meshes
			path debug_src_ply_path = out_dir_path /
				(out_file_path.stem() += "_src_mesh.ply");
			path debug_tgt_ply_path = out_dir_path /
				(out_file_path.stem() += "_tgt_mesh.ply");
			face_swap::Mesh::save_ply(fs.getSourceMesh(), debug_src_ply_path.string());
			face_swap::Mesh::save_ply(fs.getTargetMesh(), debug_tgt_ply_path.string());
		}
		if (verbose > 4)
		{
			// Write projected meshes wireframe
			path debug_src_mesh_wire_path = out_dir_path /
				(out_file_path.stem() += "_src_mesh_wire.jpg");
			cv::Mat debug_src_mesh_wire_img = fs.debugSourceMeshWireframe();
			cv::imwrite(debug_src_mesh_wire_path.string(), debug_src_mesh_wire_img);

			path debug_tgt_mesh_wire_path = out_dir_path /
				(out_file_path.stem() += "_tgt_mesh_wire.jpg");
			cv::Mat debug_tgt_mesh_wire_img = fs.debugTargetMeshWireframe();
			cv::imwrite(debug_tgt_mesh_wire_path.string(), debug_tgt_mesh_wire_img);
		}
	}
	catch (std::exception& e)
	{
		cerr << e.what() << endl;
		return 1;
	}

	return 0;
}

