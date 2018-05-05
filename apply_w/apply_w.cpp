/*
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
*/

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <cassert>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using std::cerr;
using std::cout;

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

void apply_w(const char* file_path, const cv::Mat& j1, const cv::Mat& j2)
{
    //const char* file_path = getenv("FACE_SWAP_W_PATH");
    //const char* bbox_value = getenv("FACE_SWAP_BBOX");

    if (!file_path) {
        throw std::invalid_argument("FACE_SWAP_W_PATH is not set.");
    }

    //if (!bbox_value) {
    //    throw std::invalid_argument("FACE_SWAP_BBOX is not set.");
    //}

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
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <absoluteMappings.txt> <src image> <target image>" << std::endl;
        return EXIT_FAILURE;
    }
    // Read source and target images
    const char* w_path = argv[1];
    cv::Mat source_img = cv::imread(argv[2]);
    cv::Mat target_img = cv::imread(argv[3]);

    apply_w(w_path, source_img, target_img);

	return EXIT_SUCCESS;
}

