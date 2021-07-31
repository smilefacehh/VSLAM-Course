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
    cv::Mat disp_img = cv::Mat::zeros(left_img.size(), CV_8UC1);
    
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
                double ssd_val = mango::ssd(left_patch, right_patch);
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
                disp_img.at<uchar>(r, c) = 0;
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
                    disp_img.at<uchar>(r, c) = 0;
                }
                else
                {
                    // 3个值拟合二次曲线，求极值点
                    double min_ssd_disp_left = min_ssd_disp - 1, min_ssd_disp_right = min_ssd_disp + 1;
                    double min_ssd_left = ssd_vals[min_ssd_disp_left - min_disp], min_ssd = ssd_vals[min_ssd_disp - min_disp], min_ssd_right = ssd_vals[min_ssd_disp_right - min_disp];

                    cv::Mat K = mango::polyfit(std::vector<double>{min_ssd_disp_left,(double)min_ssd_disp,min_ssd_disp_right},
                                               std::vector<double>{min_ssd_left,min_ssd,min_ssd_right},
                                               2);
                    disp_img.at<uchar>(r, c) = -K.at<double>(1,0)/(2 * K.at<double>(2,0));
                    disp_img.at<uchar>(r, c) = min_ssd_disp;
                }
            }
        }
    }

    return disp_img;
}

void Stereo::triangulate() {}
}