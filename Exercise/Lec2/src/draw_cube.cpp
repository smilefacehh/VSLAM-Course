#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camera.h"
#include "camera_mgr.h"
#include "pose.hpp"
#include "point.hpp"
#include "util.h"

static int camera_id = 0;
std::string dataFolder;
std::string confFolder;

/**
 * 世界坐标系立方体坐标点，从配置读取
*/
cv::Mat loadCube(const std::string& file)
{
    cv::Mat cube;
    cv::FileStorage fs(file, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        return cube;
    }

    fs["cube"] >> cube;
    return cube; 
}

/**
 * 在图像上绘制立方体
*/
cv::Mat drawCube(const cv::Mat& img, const mango::Pose3D& pose)
{
    cv::Mat cube = loadCube(confFolder + "camera.yaml");
    int r = cube.rows, c = cube.cols;
    assert(r == 8 && c == 3);

    cv::Mat cubeImg(img.rows, img.cols, CV_8UC3);
    if(img.channels() == 1)
    {
        std::vector<cv::Mat> channels(3, img);
        cv::merge(channels, cubeImg);
    }
    else
    {
        img.copyTo(cubeImg);
    }

    std::vector<cv::Point> points;
    for(int i = 0; i < r; i++)
    {
        double x = cube.at<double>(i, 0);
        double y = cube.at<double>(i, 1);
        double z = cube.at<double>(i, 2);
        mango::Point2D p = project(mango::Point3D(x, y, z), mango::CameraMgr::getInstance().getCameraById(camera_id), pose);
        points.push_back(cv::Point(p.x, p.y));
        cv::circle(cubeImg, cv::Point(p.x, p.y), 2, cv::Scalar(0,0,255), -1);
    }

    line(cubeImg, points[0], points[1], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[1], points[2], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[2], points[3], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[3], points[0], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[4], points[5], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[5], points[6], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[6], points[7], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[7], points[4], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[0], points[4], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[1], points[5], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[2], points[6], cv::Scalar(0,0,255), 2);
    line(cubeImg, points[3], points[7], cv::Scalar(0,0,255), 2);

    return cubeImg;
}

void run(const cv::Mat& img, const mango::Pose3D& pose)
{
    int w = img.cols, h = img.rows;
    cv::Mat undistort_img = mango::undistortImage(img, mango::CameraMgr::getInstance().getCameraById(camera_id));
    cv::Mat drawcube_img = drawCube(undistort_img, pose);

    std::vector<cv::Mat> imgs{img, drawcube_img};
    cv::Mat merge_img = mango::mergeImage(imgs, w, h);

    cv::imshow("result", merge_img);
    cv::waitKey(25);
}


int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "Usage: draw_cube data conf" << std::endl;
        return 0;
    }

    dataFolder = argv[1];
    if(dataFolder[dataFolder.length() - 1] != '/')
    {
        dataFolder.push_back('/');
    }

    confFolder = argv[2];
    if(confFolder[confFolder.length() - 1] != '/')
    {
        confFolder.push_back('/');
    }

    // 初始化camera
    static mango::CameraPtr camera_ = std::shared_ptr<mango::Camera>(new mango::Camera(++camera_id, "default camera"));
    camera_->loadConfig(confFolder + "camera.yaml");
    mango::CameraMgr::getInstance().addCamera(camera_);

    // 每帧相机位姿
    std::vector<mango::Pose3D> poses;
    try
    {
        std::string posesFile = dataFolder + "poses.txt";
        std::ifstream ifPoses(posesFile.c_str());
        if(!ifPoses.is_open())
        {
            std::cerr << "找不到文件：" << posesFile << std::endl;
            return -1;
        }
        while(!ifPoses.eof())
        {
            double wx, wy, wz, tx, ty, tz;
            ifPoses >> wx >> wy >> wz >> tx >> ty >> tz;
            poses.push_back(mango::Pose3D(wx, wy, wz, tx, ty, tz));
        }
        ifPoses.close();
    }
    catch(std::exception e)
    {
        std::cerr << "读位姿文件异常，" << e.what() << std::endl;
        return -1;
    }

    // 每帧图像数据
    try
    {
        std::string imagesFile = dataFolder + "images.txt";
        std::ifstream ifImgs(imagesFile.c_str());
        if(!ifImgs.is_open())
        {
            std::cerr << "找不到文件：" << imagesFile << std::endl;
            return -1;
        }
        int pose_idx = 0;
        while(!ifImgs.eof())
        {
            std::string imgFile;
            ifImgs >> imgFile;
            imgFile = dataFolder + "images/" + imgFile;
            cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
            run(img, poses[pose_idx++]);
        }
        ifImgs.close();
    }
    catch(std::exception e)
    {
        std::cerr << "处理图像异常，" << e.what() << std::endl;
        return -1;
    }

    return 0;
}