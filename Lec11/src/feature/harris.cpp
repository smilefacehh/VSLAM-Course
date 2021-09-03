#include "harris.h"

#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../util/math_util.h"

namespace mango {

Harris::Harris() {}
Harris::~Harris() {}

void Harris::detect(const cv::Mat& src, 
                cv::Mat& resp, 
                const int aperture_size,
                const int blockSize, 
                const double k,
                Harris::ResponseType resp_type,
                cv::BorderTypes border_type)
{
    cv::Size size = src.size();

    resp = cv::Mat::zeros(size, resp.type());

    cv::Mat dx, dy;
    cv::Sobel(src, dx, CV_32F, 1, 0, aperture_size);
    cv::Sobel(src, dy, CV_32F, 0, 1, aperture_size);

    // 创建3通道矩阵，分别存dx*dx, dx*dy, dy*dy
    cv::Mat cov(size, CV_32FC3);

    for(int i = 0; i < size.height; ++i)
    {
        float* cov_data = cov.ptr<float>(i);
        const float* dx_data = dx.ptr<float>(i);
        const float* dy_data = dy.ptr<float>(i);

        for(int j = 0; j < size.width; ++j)
        {
            float _dx = dx_data[j];
            float _dy = dy_data[j];

            cov_data[j*3] = _dx * _dx;
            cov_data[j*3+1] = _dx * _dy;
            cov_data[j*3+2] = _dy * _dy;
        }
    }

    // 方框滤波，计算M
    cv::boxFilter(cov, cov, cov.depth(), cv::Size(blockSize, blockSize), cv::Point(-1, -1), false);

    // 计算响应
    cv::Mat _resp = cv::Mat::zeros(size, CV_32FC1);
    cv::Mat _resp_norm;
    if(resp_type == HARRIS)
    {
        calcHarris(cov, _resp, k);
    }
    else if(resp_type == MINEIGENVAL)
    {
        calcMinEigenVal(cov, _resp);
    }

    // opencv接口，测试效果是否一致
    // cv::cornerHarris(src, _resp, blockSize, aperture_size, k);

    // 归一化到[0,255]
    cv::normalize(_resp, _resp_norm, 0, 255, cv::NORM_MINMAX);
    cv::convertScaleAbs(_resp_norm, resp); // 注意这里的resp变成u8类型了
}

void Harris::getKeyPoints(const cv::Mat& resp, std::vector<cv::Point2i>& kps, const int num_kps, const int nonmaximum_sup_radius)
{
    cv::Mat _resp = resp.clone();

    for(int i = 0; i < num_kps; i++)
    {
        double min_val, max_val;
        cv::Point min_idx, max_idx;
        cv::minMaxLoc(_resp, &min_val, &max_val, &min_idx, &max_idx);

        if(max_val <= 0) break;
        kps.push_back(max_idx); // 这里是图像坐标，与行列相反

        for(int j = -nonmaximum_sup_radius; j <= nonmaximum_sup_radius; j++)
        {
            for(int k = -nonmaximum_sup_radius; k <= nonmaximum_sup_radius; k++)
            {
                int r = max_idx.y + j, c = max_idx.x + k;
                if(r >= 0 && r < _resp.rows && c >= 0 && c < _resp.cols)
                {
                    _resp.at<uchar>(r, c) = 0; // 注意类型uchar
                }
            }
        }
    }
}

void Harris::getKeyPoints(const cv::Mat& resp, std::vector<cv::Point2i>& kps, const double resp_threshold, const int nonmaximum_sup_radius)
{
    cv::Mat _resp = resp.clone();
    double curr_resp = .0f;

    while(1)
    {
        double min_val;
        cv::Point min_idx, max_idx;
        cv::minMaxLoc(_resp, &min_val, &curr_resp, &min_idx, &max_idx);

        if(curr_resp < resp_threshold) break;
        kps.push_back(max_idx);

        for(int j = -nonmaximum_sup_radius; j <= nonmaximum_sup_radius; j++)
        {
            for(int k = -nonmaximum_sup_radius; k <= nonmaximum_sup_radius; k++)
            {
                int r = max_idx.y + j, c = max_idx.x + k;
                if(r >= 0 && r < _resp.rows && c >= 0 && c < _resp.cols)
                {
                    _resp.at<uchar>(r, c) = 0;
                }
            }
        }
    }
}

void Harris::getDescriptors(const cv::Mat& src, const std::vector<cv::Point2i>& kps, std::vector<std::vector<uchar>>& descriptors, const int r)
{
    int num_kp = (int)kps.size();
    descriptors.clear();
    descriptors.resize(num_kp);
    for(int i = 0; i < num_kp; i++)
    {
        descriptors[i].resize((2*r+1)*(2*r+1));
    }

    for(int i = 0; i < num_kp; i++)
    {
        int idx = 0;
        for(int j = -r; j <= r; j++)
        {
            for(int k = -r; k <= r; k++)
            {
                int row = kps[i].y + j, col = kps[i].x + k;
                if(row >= 0 && row < src.rows && col >= 0 && col < src.cols)
                {
                    descriptors[i][idx] = src.at<uchar>(row, col);
                }
                else
                {
                    descriptors[i][idx] = 0;
                }

                idx++;
            }
        }    
    }
}

void Harris::match(const std::vector<std::vector<uchar>>& reference_desc, const std::vector<std::vector<uchar>>& query_desc, std::vector<int>& match_, const double lambda)
{
    int num_kp = (int)query_desc.size();
    std::vector<double> ssd_vec(num_kp, 0);
    match_.clear();
    match_.resize(num_kp, -1);

    double global_min_ssd = std::numeric_limits<double>::max();

    for(int i = 0; i < num_kp; i++)
    {
        double min_ssd = std::numeric_limits<double>::max();
        int match_idx = -1;
        for(size_t j = 0; j < reference_desc.size(); j++)
        {
            double ssd = mango::ssd<uchar>(query_desc[i], reference_desc[j]);
            if(ssd < min_ssd)
            {
                min_ssd = ssd;
                match_idx = j;

                if(min_ssd > 0 && min_ssd < global_min_ssd)
                {
                    global_min_ssd = min_ssd;
                }
            }
        }
        ssd_vec[i] = min_ssd;
        match_[i] = match_idx;
    }

    global_min_ssd *= lambda;

    for(int i = 0; i < num_kp; i++)
    {
        if(ssd_vec[i] >= global_min_ssd)
        {
            match_[i] = -1;
        }
    }
}

cv::Mat Harris::plotMatchOneImage(const cv::Mat& query, const std::vector<cv::Point2i>& reference_kps, const std::vector<cv::Point2i>& query_kps, const std::vector<int>& match_)
{
    cv::Mat img_result(query.rows, query.cols, CV_8UC3);
    if(query.channels() == 1)
    {
        std::vector<cv::Mat> channels(3, query);
        cv::merge(channels, img_result);
    }
    else
    {
        query.copyTo(img_result);
    }

    for(int i = 0; i < match_.size(); i++)
    {
        if(match_[i] == -1) continue;

        if(match_[i] >= 0 && match_[i] < reference_kps.size())
        {
            cv::Point2i query_kp = query_kps[i], reference_kp = reference_kps[match_[i]];
            cv::line(img_result, query_kp, reference_kp, cv::Scalar(0,255,0), 2);
            cv::circle(img_result, query_kp, 3, cv::Scalar(0,0,255), -1);
        }
    }

    return img_result;
}
    
void Harris::calcMinEigenVal(const cv::Mat& cov, cv::Mat& resp)
{
    int i, j;
    cv::Size size = cov.size();

    resp = cv::Mat::zeros(size, resp.type());

    if(cov.isContinuous() && resp.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for(i = 0; i < size.height; ++i)
    {
        const float* cov_data = cov.ptr<float>(i);
        float* resp_data = resp.ptr<float>(i);

        for(j = 0; j < size.width; ++j)
        {
            float a = cov_data[j*3] * 0.5f;
            float b = cov_data[j*3+1];
            float c = cov_data[j*3+2] * 0.5f;
            resp_data[j] = (a + c) - std::sqrt((a - c)*(a - c) + b*b);
        }
    }
}

void Harris::calcHarris(const cv::Mat& cov, cv::Mat& resp, const double k)
{
    int i, j;
    cv::Size size = cov.size();
    
    resp = cv::Mat::zeros(size, resp.type());

    if(cov.isContinuous() && resp.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for(i = 0; i < size.height; ++i)
    {
        const float* cov_data = cov.ptr<float>(i);
        float* resp_data = resp.ptr<float>(i);

        for(j = 0; j < size.width; ++j)
        {
            float a = cov_data[j*3];
            float b = cov_data[j*3+1];
            float c = cov_data[j*3+2];
            resp_data[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
        }
    }
}
}