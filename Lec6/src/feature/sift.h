/**
 * SIFT
 * 1.提取几层金字塔
 * 2.每层图像，不同尺度高斯滤波
 * 3.计算DoG图像
 * 4.DoG计算极值点，提取特征点
 * 5.计算HoG描述子
 * 6.描述子匹配
*/
#pragma once

namespace mango
{

class Sift
{
public:
    Sift();
    ~Sift();

    
};

}