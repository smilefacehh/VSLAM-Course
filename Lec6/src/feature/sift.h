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

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

namespace mango
{

struct KeypointResponseGreaterThanOrEqualToThreshold
{
    KeypointResponseGreaterThanOrEqualToThreshold(float _value) :
    value(_value)
    {
    }
    inline bool operator()(const cv::KeyPoint& kpt) const
    {
        return kpt.response >= value;
    }
    float value;
};

struct KeypointResponseGreater
{
    inline bool operator()(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2) const
    {
        return kp1.response > kp2.response;
    }
};

class Sift
{
public:
    /**
     * 构造函数
     * @param nfeatures      特征点数量
     * @param octaves       octave数量
     * @param octave_scales 每个octave输出的尺度数量
     * @param contrast_threshold 每个octave输出的尺度数量
     * @param edge_threshold 每个octave输出的尺度数量
     * @param sigma          每个octave，高斯金字塔的标准差，初始值
    */
    Sift(int nfeatures, int octaves = 5, int octave_scales = 3, double contrast_threshold = 0.04, double edge_threshold = 10, double sigma = 1.6);
    ~Sift();

    /**
     * 检测
     * @param src         原始图像
     * @param keypoints   特征点
     * @param descriptors 描述子
    */
    void detect(const cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, std::vector<std::vector<float>>& descriptors);

    /**
     * 匹配
     * @param reference_desc 参考描述子
     * @param query_desc     查询描述子
     * @param match_         值存匹配上的参考描述子索引，未匹配上存-1
    */
    void match(const std::vector<std::vector<float>>& reference_desc, const std::vector<std::vector<float>>& query_desc, std::vector<int>& match_);

    /**
     * 绘制匹配图像
     * @param reference_img 参考图像
     * @param query_img     查询图像
     * @param reference_kps 参考特征点
     * @param query_kps     查询特征点
     * @param match         匹配对
     * @param saved_path    保存路径
     * @param saved         是否保存图像
    */
    void plotMatchTwoImage(const cv::Mat& reference_img, const cv::Mat& query_img, const std::vector<cv::KeyPoint>& reference_kps, const std::vector<cv::KeyPoint>& query_kps, const std::vector<int>& match, const std::string& saved_path, bool saved);

    /**
     * 绘制高斯金字塔图像
     * @param saved_path 保存路径
     * @param saved      是否保存
    */
    void plotGaussianPyramid(const std::string& saved_path, bool saved = false);

    /**
     * 绘制DoG金字塔图像
     * @param saved_path 保存路径
     * @param saved      是否保存
    */
    void plotDogPyramid(const std::string& saved_path, bool saved = false);

    /**
     * 绘制特征点
     * @param src        原始图像
     * @param kps        特征点
     * @param saved_path 保存路径
     * @param saved      是否保存
    */
    void plotKeypoints(const cv::Mat& src, const std::vector<cv::KeyPoint>& kps, const std::string& saved_path, bool saved = false);
private:
    /**
     * 构建高斯金字塔
     * @param base 原始图像放大一倍之后的图像
     * @param byr  金字塔图像集合
    */
    void buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr);

    /**
     * 创建-1对应的图像，也就是原始图像放大一倍之后的图像
     * @param img 原始图像
    */
    cv::Mat createInitialImage(const cv::Mat& img);

    /**
     * 构建DoG金字塔
     * @param gaussian_pyr 高斯金字塔
     * @param dog_pyr      DoG金字塔
    */
    void buildDogPyramid(const std::vector<cv::Mat>& gaussian_pyr, std::vector<cv::Mat>& dog_pyr);

    /**
     * 在DoG金字塔上查找极值点（8个空间邻域，9*2=18个尺度空间邻域）
     * @param dog_pyr      DoG金字塔
     * @param keypoints    特征点
    */
    void findScaleSpaceExtrema(const std::vector<cv::Mat>& dog_pyr, std::vector<cv::KeyPoint>& keypoints);

    /**
     * 保留n个响应最高的特征点
     * @param keypoints 特征点
     * @param n_points  n个点
    */
    void retainBest(std::vector<cv::KeyPoint>& keypoints, int n_points);

    /**
     * 恢复特征点的尺度，因为前面原始图像放大了一倍，这里缩放一下
     * @param keypoints 特征点
    */
    void resumeScale(std::vector<cv::KeyPoint>& keypoints);

    /**
     * 计算描述子
     * @param gaussian_pyr 高斯金字塔
     * @param kps          特征点
     * @param desc         描述子
    */
    void calcDescriptor(const std::vector<cv::Mat>& gaussian_pyr, const std::vector<cv::KeyPoint>& kps, std::vector<std::vector<float>>& desc);

    /**
     * 从patch计算描述子，1x128向量
     * @param patch_norm 16x16图像块，每个像素存梯度值
     * @param patch_dir  16x16图像块，每个像素存梯度方向
     * @param desc       描述子
    */
    void calcPatchDescriptor(const cv::Mat& patch_norm, const cv::Mat& patch_dir, std::vector<float>& desc);

    /**
     * 从cell计算HoG，1x8向量
     * @param cell_norm 4x4块，按距离加权之后的梯度值
     * @param cell_dir  4x4块，梯度方向
     * @param hog       梯度直方图
    */
    void calcCellHoG(const cv::Mat& cell_norm, const cv::Mat& cell_dir, std::vector<float>& hog);

    /**
     * 绘制金字塔图像
     * @param pyr        金字塔数组
     * @param width      初始图像宽
     * @param height     初始图像高
     * @param n          每个octave的图像数量
     * @param saved_path 保存图像路径
     * @param saved      是否保存
    */
    void plotPyramid(const std::vector<cv::Mat>& pyr, int width, int height, int n, const std::string& saved_path, bool saved);

    // 提取的特征点数量
    int nfeatures_;
    // 金字塔层级
    int octaves_;
    // 每个octave输出的scale数量，那么需要的图像数量是octave_scales_ + 3，因为n张图像会变成n-1张DoG图像，然后变成n-3个尺度
    int octave_scales_;
    double contrast_threshold_;
    double edge_threshold_;
    // 每个octave，高斯金字塔的标准差，原始图像对应的初始值
    double sigma_;

    // 高斯金字塔
    std::vector<cv::Mat> gaussian_pyr_;
    // DoG金字塔
    std::vector<cv::Mat> dog_pyr_;

    // cv::Ptr<cv::SIFT> sift_;
};

}