#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "util/image_util.h"
#include "util/string_util.h"

namespace mango
{
class KltTracker
{
public:
    KltTracker(const std::string& data_folder) : data_folder_(data_folder) {}
    ~KltTracker() {}

    void run();

private:
    void wrapTest();

    std::string data_folder_;
};

void KltTracker::run()
{
    // wrapTest();

    // std::string img_file_path = data_folder_ + "images.txt";
    // std::ifstream fs(img_file_path.c_str());
    // if(!fs.is_open())
    // {
    //     std::cerr << "can't open file:" << img_file_path << std::endl;
    //     return;
    // }

    // while(!fs.eof())
    // {
    //     std::string img_path;
    //     fs >> img_path;
    //     img_path = data_folder_ + img_path;
    //     cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    // }
    // fs.close();
}

void KltTracker::wrapTest()
{
    cv::Mat img = cv::imread("../data/000000.png", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, img.size() / 2);
    cv::Mat img_trans = warpImage(img, getWarp(50, -30, 0, 1));
    cv::Mat img_rotate = warpImage(img, getWarp(0, 0, 10 / 180.0 * M_PI, 1));
    cv::Mat img_zoom = warpImage(img, getWarp(0, 0, 0, 0.5));
    cv::Mat merge_imgs = mergeImage(std::vector<cv::Mat>{img, img_trans, img_rotate, img_zoom}, img.cols, img.rows);
    cv::imshow("merge", merge_imgs);
    cv::imwrite("../output/warp.png", merge_imgs);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "Usage: klt_track ../data" << std::endl;
        return 0;
    }

    std::string data_folder = mango::folder_add_slash(argv[1]);

    mango::KltTracker klt_tracker(data_folder);
    klt_tracker.run();
    return 0;
}