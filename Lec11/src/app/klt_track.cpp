#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vo/klt.h"
#include "util/image_util.h"
#include "util/string_util.h"
#include "util/math_util.h"
#include "util/draw_util.h"

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
    void wrapAndTrackTest();
    void trackPointKLTTest();

    std::string data_folder_;
};

void KltTracker::run()
{
    // wrapTest();
    // wrapAndTrackTest();
    // trackPointKLTTest();

    KanadeLucasTomasi klt;
    std::vector<cv::Point2d> kp_prev;

    // 读取关键点坐标
    std::string kp_file_path = data_folder_ + "keypoints.txt";
    std::ifstream fs(kp_file_path.c_str());
    if(!fs.is_open())
    {
        std::cerr << "can't open file:" << kp_file_path << std::endl;
        return;
    }
    while(!fs.eof())
    {
        double x, y;
        fs >> y >> x;
        if(fs.fail()) break;
        kp_prev.push_back(cv::Point2d(x, y));
    }
    fs.close();

    // 读取图像
    std::string img_file_path = data_folder_ + "images.txt";
    fs.open(img_file_path.c_str());
    if(!fs.is_open())
    {
        std::cerr << "can't open file:" << img_file_path << std::endl;
        return;
    }

    cv::Mat img_prev;

    int i = 0;
    while(!fs.eof())
    {
        std::string img_path;
        fs >> img_path;
        img_path = data_folder_ + img_path;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

        if(i == 0)
        {
            img_prev = img;
            cv::Mat img_kp = drawPoint<cv::Point2d>(img, kp_prev, DrawType::X, cv::Scalar(0,0,255));
            cv::imshow("1", img_kp);
            cv::waitKey(-1);
            cv::destroyAllWindows();
        }
        else
        {
            std::vector<cv::Point2d> kp(kp_prev.size());
            for(int j = 0; j < kp_prev.size(); j++)
            {
                Eigen::Matrix<float, 2, 3> warp_transform = klt.trackPointKLT(img_prev, img, kp_prev[j].x, kp_prev[j].y, 15, 50);
                kp[j] = cv::Point2d(kp_prev[j].x + warp_transform(0,2), kp_prev[j].y + warp_transform(1,2));
            }

            // cv::Mat img_prev_kp = drawPoint<cv::Point2d>(img_prev, kp_prev, DrawType::X, cv::Scalar(0,0,255));
            cv::Mat img_kp = drawPoint<cv::Point2d>(img, kp, DrawType::X, cv::Scalar(0,0,255));
            for(int j = 0; j < kp.size(); j++)
            {
                cv::line(img_kp, kp_prev[j], kp[j], cv::Scalar(0,255,0));
            }
            cv::imshow("1", img_kp);
            cv::waitKey(-1);
            cv::destroyAllWindows();
        }

        i++;
    }
    fs.close();
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

// 在参考图像中某个位置取一个patch，然后将图像做一个形变（平移）处理得到查询图像，在同样的位置附近取很多patch
// patch计算ssd，取最小值的patch，从而找到匹配点
void KltTracker::wrapAndTrackTest()
{
    float x = 900; float y = 291;
    float patch_radius = 15;
    cv::Mat img = cv::imread("../data/000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat patch = getWarpedPatch(img, getWarp(0, 0, 0, 1), x, y, patch_radius);

    cv::Mat img_query = warpImage(img, getWarp(10, 6, 0, 1));
    const int search_radius = 20;
    cv::Mat ssds = cv::Mat::zeros(cv::Size(2*search_radius+1, 2*search_radius+1), CV_32FC1);
    for(int i = -search_radius; i <= search_radius; i++)
    {
        for(int j = -search_radius; j <= search_radius; j++)
        {
            cv::Mat patch_query = getWarpedPatch(img_query, getWarp(i, j, 0, 1), x, y, patch_radius);
            double ssd_val = ssd<uchar>(patch, patch_query);
            ssds.at<float>(i + search_radius, j + search_radius) = ssd_val;
        }
    }

    cv::Mat merge_imgs = mergeImage(std::vector<cv::Mat>{gray2color(patch), gray2color(ssds)}, ssds.cols, ssds.rows);
    cv::imshow("merge", merge_imgs);
    cv::imwrite("../output/warp_bruteforce.png", merge_imgs);
    cv::waitKey(-1);
    cv::destroyAllWindows();
}

void KltTracker::trackPointKLTTest()
{
    KanadeLucasTomasi klt;
    cv::Mat img = cv::imread("../data/000000.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_query = warpImage(img, getWarp(8, 6, 0, 1));
    Eigen::Matrix<float, 2, 3> warp_transform = klt.trackPointKLT(img, img_query, 900, 291, 15, 50);
    std::cout << warp_transform << std::endl;
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