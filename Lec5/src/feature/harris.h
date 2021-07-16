#pragma once

#include <opencv2/opencv.hpp>

namespace mango {

class Harris
{
public:

    enum ResponseType
    {
        HARRIS,
        MINEIGENVAL
    };

    Harris();
    ~Harris();

    /**
     * harris角点检测，特征点数量
     * @param src                   源图像
     * @param resp                  角点响应矩阵，[0,255],uchar，值越大表示响应越大
     * @param aperture_size         sobel算子大小
     * @param blockSize             邻域窗边长
     * @param k                     魔数k
     * @param resp_type             计算响应的方式，harris、shi-tomasi
     * @param border_type           图像边界复制形式
    */
    void detect(const cv::Mat& src, 
                cv::Mat& resp, 
                const int aperture_size,
                const int blockSize, 
                const double k,
                Harris::ResponseType resp_type = HARRIS,
                cv::BorderTypes border_type = cv::BORDER_DEFAULT);

    /**
     * 按数量提取特征点
     * @param resp                  角点响应图像
     * @param kps                   特征点图像坐标，注意跟行列反过来了
     * @param num_kps               最多提取特征点数量
     * @param nonmaximum_sup_radius 响应点非极大值抑制范围半径
    */
    void getKeyPoints(const cv::Mat& resp, std::vector<cv::Point2i>& kps, const int num_kps, const int nonmaximum_sup_radius = 8);

    /**
     * 按阈值提取特征点
     * @param resp                  角点响应图像
     * @param kps                   特征点图像坐标，注意跟行列反过来了
     * @param resp_threshold        响应阈值
     * @param nonmaximum_sup_radius 响应点非极大值抑制范围半径
    */
    void getKeyPoints(const cv::Mat& resp, std::vector<cv::Point2i>& kps, const double resp_threshold, const int nonmaximum_sup_radius = 8);
   
    /**
     * 计算描述子
     * @param src         原始图像
     * @param kps         关键点
     * @param descriptors 描述子，N*(2r+1)^2，N个关键点，用半径r的邻域像素灰度值构成的向量来表示描述子
     * @param r           方框半径
    */
    void getDescriptors(const cv::Mat& src, const std::vector<cv::Point2i>& kps, std::vector<std::vector<uchar>>& descriptors, const int r);

    /**
     * 描述子匹配
     * @param reference_desc 参考描述子
     * @param query_desc     待匹配描述子
     * @param match_         size与query_desc一致，存匹配上的reference_desc对应的索引，未匹配上的存-1
     * @param lambda         SSD >= lambda*min(SSD)，则认为没有匹配上
    */
    void match(const std::vector<std::vector<uchar>>& reference_desc, const std::vector<std::vector<uchar>>& query_desc, std::vector<int>& match_, const double lambda);
    
    /**
     * 在一张图像上绘制匹配（query上）
     * @param query         匹配图像
     * @param reference_kps 参考特征点
     * @param query_kps     匹配特征点
     * @param match_        size与query_kps一致，存匹配上的reference_kps对应的索引，未匹配上的存-1
    */
    cv::Mat plotMatchOneImage(const cv::Mat& query, const std::vector<cv::Point2i>& reference_kps, const std::vector<cv::Point2i>& query_kps, const std::vector<int>& match_);

    cv::Mat plotMatchTwoImage();

private:
    /**
     * shi-tomasi响应
     * @param cov  M矩阵
     * @param resp 响应
    */
    void calcMinEigenVal(const cv::Mat& cov, cv::Mat& resp);

    /**
     * harri 响应
     * @param cov  M矩阵
     * @param resp 响应
     * @param k    魔数
    */
    void calcHarris(const cv::Mat& cov, cv::Mat& resp, const double k);

};

}