#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "util/string_util.h"
#include "util/image_util.h"
#include "vo/stereo.h"

namespace mango {

class StereoVision
{
public:
    StereoVision(const std::string& data_folder)
        : data_folder_(data_folder)
    {}
    ~StereoVision() {}

    void run();

private:
    std::string data_folder_;
};

void StereoVision::run()
{
    try
    {
        // std::string imgfile = data_folder_+ "images.txt";
        // std::ifstream ifs(imgfile.c_str());

        cv::Mat left_img = cv::imread(data_folder_ + "left/000000.png", cv::IMREAD_GRAYSCALE);
        cv::Mat right_img = cv::imread(data_folder_ + "right/000000.png", cv::IMREAD_GRAYSCALE);
        mango::Stereo stereo;
        cv::Mat disp_img = stereo.match(left_img, right_img, 5, 5, 50);
        cv::imshow("1", mango::gray2color(disp_img));
        cv::waitKey(-1);
        cv::destroyWindow("1");
    }
    catch(std::exception e)
    {
        std::cerr << "image process exception: " << e.what() << std::endl;
    }
}

}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "Usage: stereo_vision ../data" << std::endl;
        return 0;
    }

    std::string data_folder = mango::folder_add_slash(argv[1]);

    mango::StereoVision stereo_vision(data_folder);
    stereo_vision.run();

    return 0;
}