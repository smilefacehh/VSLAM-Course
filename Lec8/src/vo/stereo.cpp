#include "stereo.h"

#include <limits.h>
#include <omp.h>

#include "util/math_util.h"


namespace mango
{
Stereo::Stereo() {}
Stereo::~Stereo() {}

cv::Mat Stereo::match(const cv::Mat& left_img, const cv::Mat& right_img, float patch_radius, float min_disp, float max_disp)
{
    cv::Mat disp_img = cv::Mat::zeros(left_img.size(), CV_32FC1);
    
    int rmax = left_img.rows - (int)patch_radius, cmax = left_img.cols - (int)patch_radius;

    // 遍历左图像素点
#pragma omp parallel for
    for(int r = patch_radius; r < rmax; r++)
    {
        int rstart = r-patch_radius, rend = r+patch_radius+1;

#pragma omp parallel for
        for(int c = max_disp + patch_radius; c < cmax; c++)
        {
            int cstart = c-patch_radius, cend = c+patch_radius+1;

            // 所有候选滑窗的ssd值
            std::vector<double> ssd_vals(max_disp - min_disp + 1, 0);

            // 最小的ssd值
            double min_ssd_val = std::numeric_limits<double>::max();
            // 最小的ssd值对应的视差
            int min_ssd_disp = -1;
            
            cv::Mat left_patch = left_img.rowRange(rstart, rend).colRange(cstart, cend); // 定义放里面，放外面不行
            // 遍历右图扫描线，范围
            for(int i = min_disp; i <= max_disp; i++)
            {
                cv::Mat right_patch = right_img.rowRange(rstart, rend).colRange(cstart-i, cend-i);
                double ssd_val = mango::ssd<uchar>(left_patch, right_patch);
                ssd_vals[i - min_disp] = ssd_val;
                if(ssd_val < min_ssd_val)
                {
                    min_ssd_val = ssd_val;
                    min_ssd_disp = i;
                }
            }
            
            // 边界上的点也不考虑，因为后面要用相邻两个点拟合二次曲线，找最低点，亚像素插值
            if(min_ssd_disp == -1 || min_ssd_disp == min_disp || min_disp == max_disp)
            {
                disp_img.at<float>(r, c) = 0;
            }
            else
            {
                // 超过3个ssd比较小的点，这个点认为不够准，视差设为0
                int count = 0;
                int th = 3;
                for(int i = 0; i < max_disp - min_disp + 1; i++)
                {
                    if(ssd_vals[i] <= 1.5 * min_ssd_val)
                    {
                        count++;
                        if(count >= th)
                        {
                            break;
                        }
                    }
                }
                if(count >= th)
                {
                    disp_img.at<float>(r, c) = 0;
                }
                else
                {
                    // 3个值拟合二次曲线，求极值点
                    double min_ssd_disp_left = min_ssd_disp - 1, min_ssd_disp_right = min_ssd_disp + 1;
                    double min_ssd_left = ssd_vals[min_ssd_disp_left - min_disp], min_ssd = ssd_vals[min_ssd_disp - min_disp], min_ssd_right = ssd_vals[min_ssd_disp_right - min_disp];

                    cv::Mat K = mango::polyfit(std::vector<double>{min_ssd_disp_left,(double)min_ssd_disp,min_ssd_disp_right},
                                               std::vector<double>{min_ssd_left,min_ssd,min_ssd_right},
                                               2);

                    if(K.at<double>(2,0) <= 0 || abs(-K.at<double>(1,0)/(2 * K.at<double>(2,0)) - min_ssd_disp) >= 1)
                    {
                        disp_img.at<float>(r, c) = min_ssd_disp;
                    }
                    else
                    {
                        disp_img.at<float>(r, c) = -K.at<double>(1,0)/(2 * K.at<double>(2,0));
                    }
                }
            }
        }
    }

    return disp_img;
}

pcl::PointCloud<pcl::PointXYZRGB> Stereo::disparity2pointcloud(const cv::Mat& disparity, const cv::Mat& img, cv::Mat& depth, const Eigen::Matrix3d& K, double baseline)
{
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
    // 临时存
    cv::Mat postmp = cv::Mat(disparity.size(), CV_32FC3);
    depth = cv::Mat(disparity.size(), CV_32FC1);

#pragma omp parallel for
    for(int r = 0; r < disparity.rows; r++)
    {
#pragma omp parallel for
        for(int c = 0; c < disparity.cols; c++)
        {
            if(disparity.at<float>(r, c) > 0)
            {
                Eigen::Vector3d p0(r, c, 1);
                Eigen::Vector3d p1(r, c - disparity.at<float>(r,c), 1);
                Eigen::Vector3d p0_ = K.inverse() * p0;
                Eigen::Vector3d p1_ = K.inverse() * p1;
                Eigen::Matrix<double, 3, 2> A;
                A.block(0, 0, 3, 1) = p0_;
                A.block(0, 1, 3, 1) = p1_;
                Eigen::Vector3d b(baseline, 0, 0);
                Eigen::Vector2d lambda = (A.transpose() * A).inverse() * (A.transpose() * b);
                Eigen::Vector3d P = lambda(0) * K.inverse() * p0;
                depth.at<float>(r, c) = P(2);
                postmp.at<cv::Vec3f>(r, c)[0] = P(0);
                postmp.at<cv::Vec3f>(r, c)[1] = P(1);
                postmp.at<cv::Vec3f>(r, c)[2] = P(2);
            }
        }
    }

    // 并行for里面不能用push_back
    for(int r = 0; r < postmp.rows; r++)
    {
        for(int c = 0; c < postmp.cols; c++)
        {
            if(disparity.at<float>(r,c) > 0)
            {
                pcl::PointXYZRGB p;
                p.x = postmp.at<cv::Vec3f>(r, c)[0];
                p.y = postmp.at<cv::Vec3f>(r, c)[1];
                p.z = postmp.at<cv::Vec3f>(r, c)[2];
                if(img.channels() == 3)
                {
                    p.b = img.at<cv::Vec3f>(r, c)[0];
                    p.g = img.at<cv::Vec3f>(r, c)[1];
                    p.r = img.at<cv::Vec3f>(r, c)[2];
                }
                else
                {
                    p.b = img.at<uchar>(r, c);
                    p.g = p.b;
                    p.r = p.b;
                }
                pointcloud.push_back(p);
            }
        }
    }
    return pointcloud;
}

}