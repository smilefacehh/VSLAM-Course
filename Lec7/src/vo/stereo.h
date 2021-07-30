#ifndef STEREO_H_
#define STEREO_H_

#include <opencv2/opencv.hpp>

/**
 * Stereo双目
 * 
 * 1.输入已经校正之后的双目图像，查找匹配点，输出视差图，并行滑窗匹配，SSD
 * 2.剔除外点
 * 3.三角化计算3D点
 * 4.利用外参，累积点云，展示
*/

namespace mango
{
class Stereo
{
public:
    Stereo();
    ~Stereo();

    /**
     * 
    */
    cv::Mat match(const cv::Mat& left_img, const cv::Mat& right_img, float patch_radius, float min_disp, float max_disp);
    void triangulate();

};
}
#endif