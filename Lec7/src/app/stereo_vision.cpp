#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

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
        double baseline = 0.54; //KITTI
        Eigen::Matrix3d K; // 相机内参

        std::string intrinsic_file = data_folder_ + "K.txt";
        std::ifstream intrinsic_ifs(intrinsic_file.c_str());
        if(!intrinsic_ifs.is_open())
        {
            std::cout << "open file intrinsics failed" << std::endl;
            return;
        }
        intrinsic_ifs >> K(0,0) >> K(0,1) >> K(0,2) >> K(1,0) >> K(1,1) >> K(1,2) >> K(2,0) >> K(2,1) >> K(2,2);

        std::string imgfile = data_folder_+ "images.txt";
        std::ifstream imgifs(imgfile.c_str());

        std::string posefile = data_folder_ + "poses.txt";
        std::ifstream poseifs(posefile.c_str());

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZRGB>());
        int i = 0;
        while(!imgifs.eof() && !poseifs.eof())
        {
            auto t_start = std::chrono::steady_clock::now();
            // 读取双目图像
            std::string left_img_name, right_img_name;
            imgifs >> left_img_name >> right_img_name;

            if(left_img_name.empty() || right_img_name.empty()) break;

            left_img_name = data_folder_ + "left/" + left_img_name;
            right_img_name = data_folder_ + "right/" + right_img_name;

            cv::Mat left_img = cv::imread(left_img_name, cv::IMREAD_GRAYSCALE);
            cv::Mat right_img = cv::imread(right_img_name, cv::IMREAD_GRAYSCALE);

            // 读取pose（左目）
            Eigen::Matrix<double, 3, 4> T_wc = Eigen::Matrix<double, 3, 4>::Zero();
            for(int i = 0; i < 12; i++)
            {
                poseifs >> T_wc(i / 4, i % 4);
            }

            // 计算双目视差
            mango::Stereo stereo;
            cv::Mat disp_img = stereo.match(left_img, right_img, 5, 5, 50);

            // cv::Mat disp_color = mango::gray2color(disp_img);
            // cv::imshow("1", disp_color);
            // cv::waitKey(-1);
            // cv::destroyWindow("1");

            // 视差计算3D坐标
            cv::Mat depth;
            pcl::PointCloud<pcl::PointXYZRGB> cloud = stereo.disparity2pointcloud(disp_img, left_img, depth, K, baseline);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudptr(new pcl::PointCloud<pcl::PointXYZRGB>(cloud));

            // cv::Mat depthcolor = mango::gray2color(depth);
            // cv::imshow("2", depthcolor);
            // cv::waitKey(-1);
            // cv::destroyWindow("2");

            // 坐标转换
            Eigen::Matrix3d R_cf;
            R_cf << 0, -1,  0,
                    0,  0, -1,
                    1,  0,  0;
            for(int i = 0; i < cloudptr->points.size(); i++)
            {
                Eigen::Vector3d p_c(cloudptr->points[i].x, cloudptr->points[i].y, cloudptr->points[i].z);
                Eigen::Vector3d p_f = R_cf.inverse() * p_c;
                Eigen::Matrix<double, 4, 4> R_tmp = Eigen::Matrix<double, 4, 4>::Zero();
                R_tmp.block(0,0,3,3) = R_cf;
                R_tmp(3,3) = 1;
                Eigen::Matrix<double, 3, 4> T_wf = T_wc * R_tmp;
                Eigen::Vector3d p_w = T_wf.block(0,0,3,3) * p_f + T_wf.block(0,3,3,1);
                cloudptr->points[i].x = p_w(0);
                cloudptr->points[i].y = p_w(1);
                cloudptr->points[i].z = p_w(2);
            }

            // 点云累加
            *cloud_all += *cloudptr;
            
            // 点云显示
            // std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            // viewer->setBackgroundColor(0, 0, 0);
            // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_all);
            // viewer->addPointCloud<pcl::PointXYZRGB>(cloud_all, rgb, "sample cloud");
            // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud"); //变量名，赋值
            // viewer->addCoordinateSystem(1.0);
            // viewer->initCameraParameters();

            // while (!viewer->wasStopped())
            // {
            //     viewer->spinOnce(1000);
            // }

            auto t_end = std::chrono::steady_clock::now();
            auto t_diff = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start);
            std::cout << ++i << "/100, time cost:" << t_diff.count() << "s" << std::endl;
        }
        pcl::io::savePCDFileASCII("../output/cloud.pcd", *cloud_all);
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