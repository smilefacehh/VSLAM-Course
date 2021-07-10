#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/eigen.hpp>

#include <eigen3/Eigen/Core>

#include "../util/string_util.h"
#include "../util/file_util.h"
#include "../util/util.h"
#include "../util/draw_util.h"
#include "../base/camera.h"
#include "../base/camera_mgr.h"
#include "../base/pose.hpp"
#include "../calib/dlt.h"
#include "../base/point.hpp"

class Calib
{
public:
    Calib(const std::string& data_folder, const std::string& conf_folder);
    ~Calib();

    void init();
    void run();

private:
    void runDLT(const Eigen::Matrix<double, Eigen::Dynamic, 3>& world_pts, 
                const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs,
                mango::Pose3D& pose, 
                Eigen::Matrix3d& camera_K);

    std::string data_foler_, conf_folder_; // 数据、配置路径
    int camera_id;                         // 相机ID
    Eigen::MatrixXd world_pts_;            // 角点世界坐标
    Eigen::MatrixXd corner_pxs;            // 角点投影像素坐标
};

Calib::Calib(const std::string& data_folder, const std::string& conf_folder)
    : data_foler_(data_folder), conf_folder_(conf_folder), camera_id(0)
{}
Calib::~Calib() {}

/**
 * 初始化
*/
void Calib::init()
{
    // 读取角点的世界坐标，注意单位，要乘上0.01!!
    std::string camera_config_path = conf_folder_ + "camera.yaml";
    cv::FileStorage fs(camera_config_path, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        std::cerr << "failed to load camera.yaml" << std::endl;
        return;
    }
    cv::Mat corners;
    fs["corners"] >> corners;

    world_pts_.resize(corners.rows, corners.cols);
    for(int i = 0; i < corners.rows; i++)
    {
        for(int j = 0; j < corners.cols; j++)
        {
            world_pts_(i, j) = corners.at<double>(i, j) * 0.01;
        }
    }

    // 读取角点的投影像素坐标
    std::string project_px_path = data_foler_ + "detected_corners.txt";
    corner_pxs = mango::load_matrix(project_px_path, 120, 24);

    // 初始化camera
    static mango::CameraPtr camera_ = std::shared_ptr<mango::Camera>(new mango::Camera(++camera_id, "default camera"));
    camera_->loadConfig(conf_folder_ + "camera.yaml");
    mango::CameraMgr::getInstance().addCamera(camera_);
}

/**
 * 主流程
*/
void Calib::run()
{
    // 读取图像数据
    // 调用DLT求解M。内参已知、内参未知（看看内参误差）
    // 调用PNP求解。效率对比
    // M分解得到R、t
    // 在图像中绘制pose
    // 绘制给出的投影点，画圆；计算的pose投影点画x
    try
    {
        std::string img_file_path = data_foler_ + "images.txt";
        std::ifstream fs(img_file_path.c_str());
        if(!fs.is_open())
        {
            std::cerr << "找不到文件：" << img_file_path << std::endl;
            return;
        }

        int id = 0;
        while(!fs.eof() && id < corner_pxs.rows())
        {
            std::string img_path;
            fs >> img_path;

            img_path = data_foler_ + "images_undistorted/" + img_path;
            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

            // 一帧世界点-投影点
            Eigen::MatrixXd px_line = corner_pxs.row(id++);
            Eigen::MatrixXd pxs;
            pxs.resize(px_line.size() / 2, 2);
            for(int i = 0; i < px_line.size() / 2; i++)
            {
                pxs(i, 0) = px_line(2 * i);
                pxs(i, 1) = px_line(2 * i + 1);
            }
            // std::cout << "pxs:" << pxs << std::endl;
            // DLT计算外参
            mango::Pose3D pose;
            auto& camera = mango::CameraMgr::getInstance().getCameraById(camera_id);
            Eigen::Matrix3d camera_K = camera->K();
            runDLT(world_pts_, pxs, pose, camera_K);
            
            // 给定像素点画圆，根据外参投影得到的像素点画叉叉
            Eigen::Matrix<double, Eigen::Dynamic, 2> reproject_pxs;
            reproject_pxs.resize(world_pts_.rows(), 2);

            for(int i = 0; i < world_pts_.rows(); i++)
            {
                mango::Point3D pt(world_pts_(i, 0), world_pts_(i, 1), world_pts_(i, 2));
                mango::Point2D px = mango::project(pt, camera, pose);
                reproject_pxs(i, 0) = px.x;
                reproject_pxs(i, 1) = px.y;
            }

            // mango::Point3D pose_pt = mango::Point3D(pose.origin());
            // mango::Point2D pose_px = mango::project(pose_pt, camera, pose);

            cv::Mat img_1 = mango::drawPoint(img, pxs, mango::DrawType::CIRCLE, cv::Scalar(0, 255, 0));
            cv::Mat img_2 = mango::drawPoint(img_1, reproject_pxs, mango::DrawType::X, cv::Scalar(0, 0, 255));
            // cv::Mat img_3 = mango::drawPoint(img_2, std::vector<mango::Point2D>{pose_px}, mango::DrawType::RECT, cv::Scalar(0, 0, 255));
            cv::imshow("result", img_2);
            cv::waitKey(-1);
        }
        fs.close();
    }
    catch(std::exception e)
    {
        std::cerr << "处理图像异常，" << e.what() << std::endl;
    }
}

void Calib::runDLT(const Eigen::Matrix<double, Eigen::Dynamic, 3>& world_pts, 
                const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs,
                mango::Pose3D& pose, 
                Eigen::Matrix3d& camera_K)
{
    mango::DLT dlt(world_pts, pxs, camera_K);
    dlt.run();
    pose = dlt.getPose();
    camera_K = dlt.getCameraK();
    // std::cout << "camera_K:\n" << camera_K << std::endl;;
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "Usage: calib data conf" << std::endl;
        return 0;
    }

    std::string data_folder = mango::folder_add_slash(argv[1]);
    std::string conf_folder = mango::folder_add_slash(argv[2]);

    Calib calib(data_folder, conf_folder);
    calib.init();
    calib.run();
   
    return 0;
}