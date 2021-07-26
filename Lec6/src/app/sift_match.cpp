#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "feature/sift.h"
#include "util/string_util.h"
#include "util/image_util.h"

namespace mango {

class SiftMatch
{
public:
    SiftMatch(const std::string& data_folder)
        : data_folder_(data_folder)
    {}
    ~SiftMatch() {}

    void run();

private:
    std::string data_folder_;
};

void SiftMatch::run()
{
    try
    {
        mango::Sift sift(1000, 5, 3, 125, 10, 1.6);
        cv::Mat image1 = cv::imread(data_folder_ + "img_1.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat image2 = cv::imread(data_folder_ + "img_2.jpg", cv::IMREAD_GRAYSCALE);

        std::vector<cv::KeyPoint> kps;
        std::vector<std::vector<float>> desc;
        sift.detect(image1, kps, desc);
        // sift.plotGaussianPyramid("/home/lutao/workspace/slam/VSLAM-Course/Lec6/output/gaussian_pyr.png", true);
        // sift.plotDogPyramid("/home/lutao/workspace/slam/VSLAM-Course/Lec6/output/dog_pyr.png", true);
        sift.plotKeypoints(image1, kps);

        // cv::Mat merged_image = mango::mergeImage(std::vector<cv::Mat>{image1, image2}, image1.cols/5, image1.rows/5);
        // cv::imshow("match", merged_image);
        // cv::waitKey(-1);
        // cv::destroyAllWindows();
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
        std::cout << "Usage: sift_match ../data" << std::endl;
        return 0;
    }

    std::string data_folder = mango::folder_add_slash(argv[1]);

    mango::SiftMatch sift_match(data_folder);
    sift_match.run();

    return 0;
}