#pragma once

#include <opencv2/opencv.hpp>

namespace mango
{
/**
 * 合并图像
 * @param imgs 图像
 * @param w    图像缩放为指定大小，宽
 * @param h    图像缩放为指定大小，高
*/
cv::Mat mergeImage(const std::vector<cv::Mat>& imgs, int w, int h);

/**
 * 灰度图映射成彩色图，显示用
 * @param img 灰度图像
*/
cv::Mat gray2color(const cv::Mat& img);
}