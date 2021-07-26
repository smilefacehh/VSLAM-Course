#include "sift.h"

#include "util/draw_util.h"


namespace mango
{
Sift::Sift(int nfeatures, int octaves, int octave_scales, double contrast_threshold, double edge_threshold, double sigma)
    : nfeatures_(nfeatures), octaves_(octaves), octave_scales_(octave_scales), contrast_threshold_(contrast_threshold), edge_threshold_(edge_threshold), sigma_(sigma)
{
    sift_ = cv::SIFT::create(100);
}

Sift::~Sift() {}

void Sift::detect(const cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, std::vector<std::vector<float>>& descriptors)
{
    cv::Mat base = createInitialImage(src);
    buildGaussianPyramid(base, gaussian_pyr_);
    buildDogPyramid(gaussian_pyr_, dog_pyr_);

    findScaleSpaceExtrema(dog_pyr_, keypoints);
    std::cout << keypoints.size() << std::endl;
    retainBest(keypoints, nfeatures_);
    std::cout << keypoints.size() << std::endl;
    resumeScale(keypoints);
    std::cout << keypoints.size() << std::endl;
}

void Sift::plotGaussianPyramid(const std::string& saved_path, bool saved)
{
    if(gaussian_pyr_.empty())
    {
        return;
    }
    
    int w = gaussian_pyr_[0].cols, h = gaussian_pyr_[0].rows;
    int n = gaussian_pyr_.size() / octaves_;
    plotPyramid(gaussian_pyr_, w / 30, h / 30, n, saved_path, saved);
}

void Sift::plotDogPyramid(const std::string& saved_path, bool saved)
{
    if(dog_pyr_.empty())
    {
        return;
    }

    std::vector<cv::Mat> dog_pyr_norm(dog_pyr_.size());
    for(size_t i = 0; i < dog_pyr_.size(); i++)
    {
        cv::normalize(dog_pyr_[i], dog_pyr_norm[i], 0, 255, cv::NORM_MINMAX);
    }

    int w = dog_pyr_norm[0].cols, h = dog_pyr_norm[0].rows;
    int n = dog_pyr_norm.size() / octaves_;
    plotPyramid(dog_pyr_norm, w / 30, h / 30, n, saved_path, saved);
}

void Sift::plotPyramid(const std::vector<cv::Mat>& pyr, int width, int height, int n, const std::string& saved_path, bool saved)
{
    int octave = pyr.size() / n;
    cv::Mat pyr_img((2 - std::pow(2, -(octave - 1))) * height, n * width, CV_8UC1);

    int w = width, h = height;
    for(int o = 0; o < octave; o++)
    {
        for(int i = 0; i < n; i++)
        {
            const cv::Mat& img = pyr[o * n + i];
            cv::Mat tmp;
            cv::resize(img, tmp, cv::Size(w, h));
            tmp.copyTo(pyr_img(cv::Rect(w * i, (2 - std::pow(2, -(o - 1))) * height, w, h)));
        }
        w *= 0.5;
        h *= 0.5;
    }
    if(saved)
    {
        cv::imwrite(saved_path, pyr_img);
    }
    cv::imshow("pyramid", pyr_img);
    cv::waitKey(-1);
    cv::destroyWindow("pyramid");
}

void Sift::plotKeypoints(const cv::Mat& src, const std::vector<cv::KeyPoint>& kps)
{
    std::cout << kps.size() << std::endl;
    for(size_t i = 0; i < kps.size(); i++)
    {
        std::cout << kps[i].pt.x << "," << kps[i].pt.y << "," << kps[i].size << std::endl;
    }
    cv::Mat img_kp = mango::drawKeyPoint(src, kps, cv::Scalar(0, 0, 255));
    cv::resize(img_kp, img_kp, cv::Size(img_kp.cols / 5, img_kp.rows / 5));
    cv::imshow("kp", img_kp);
    cv::waitKey(-1);
    cv::destroyWindow("kp");
}

void Sift::buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr)
{
    pyr.clear();
    pyr.resize(octaves_ * (octave_scales_ + 3));

    // 每个octave，高斯滤波参数
    // todo 为什么是这个公式 \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2， 跟ppt中的不一样
    std::vector<double> sig(octave_scales_ + 3);
    sig[0] = sigma_;
    double k = std::pow(2.0, 1.0 / octave_scales_);
    for(int i = 1; i < octave_scales_ + 3; i++)
    {
        double sig_prev = std::pow(k, (double)(i - 1)) * sigma_;
        double sig_total = sig_prev * k;
        sig[i] = std::sqrt(sig_total * sig_total - sig_prev * sig_prev);
        // std::cout << sig[i] << std::endl;
    }

    // std::cout << "octaves_:" << octaves_ << ", n:" << octave_scales_ + 3 << std::endl;

    for(int o = 0; o < octaves_; o++)
    {
        for(int i = 0; i < octave_scales_ + 3; i++)
        {
            cv::Mat& dst = pyr[o * (octave_scales_ + 3) + i];
            if(o == 0 && i == 0)
            {
                dst = base;
            }
            else if(i == 0)
            {
                // todo 为什么下一层octave第一个图像是用上一层octave的倒数第2个来减半
                const cv::Mat& src = pyr[(o - 1) * (octave_scales_ + 3) + octave_scales_];
                cv::resize(src, dst, cv::Size(src.cols/2, src.rows/2), 0, 0, cv::INTER_NEAREST);
            }
            else
            {
                const cv::Mat& src = pyr[o * (octave_scales_ + 3) + i - 1];
                cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }
}

cv::Mat Sift::createInitialImage(const cv::Mat& img)
{
    cv::Mat upper_img, base;
    cv::resize(img, upper_img, cv::Size(img.cols * 2, img.rows * 2), 0, 0, cv::INTER_LINEAR);
    float sig_diff = std::pow(std::max(sigma_ * sigma_ - 0.5 * 0.5 * 4, 0.01), 0.5);
    cv::GaussianBlur(upper_img, base, cv::Size(), sig_diff, sig_diff);
    return base;
}

void Sift::buildDogPyramid(const std::vector<cv::Mat>& gaussian_pyr, std::vector<cv::Mat>& dog_pyr)
{
    dog_pyr.clear();
    dog_pyr.resize(octaves_ * (octave_scales_ + 2));

    for(int idx = 0; idx < int(dog_pyr.size()); idx++)
    {
        int o = idx / (octave_scales_ + 2);
        int i = idx % (octave_scales_ + 2);

        const cv::Mat& src1 = gaussian_pyr[o * (octave_scales_ + 3) + i];
        const cv::Mat& src2 = gaussian_pyr[o * (octave_scales_ + 3) + i + 1];
        cv::Mat& dst = dog_pyr[o * (octave_scales_ + 2) + i];
        // cv::Mat tmp;
        cv::subtract(src2, src1, dst);
        // cv::normalize(tmp, dst, 0, 255, cv::NORM_MINMAX);
    }
}

void Sift::findScaleSpaceExtrema(const std::vector<cv::Mat>& dog_pyr, std::vector<cv::KeyPoint>& keypoints)
{
    // 边界不提取特征点
    int border = 5;
    for(int o = 0; o < octaves_; o++)
    {
        for(int i = 1; i <= octave_scales_; i++)
        {
            int idx = o * (octave_scales_ + 2) + i;
            const cv::Mat& img = dog_pyr[idx];
            const cv::Mat& prev = dog_pyr[idx - 1];
            const cv::Mat& next = dog_pyr[idx + 1];

            for(int r = border; r < img.rows - border; r++)
            {
                for(int c = border; c < img.cols - border; c++)
                {
                    float val = img.at<float>(r, c);
                    if(std::abs(val) <= contrast_threshold_)
                    {
                        continue;
                    }
                    float _00, _01, _02;
                    float _10,      _12;
                    float _20, _21, _22;
                    _00 = img.at<float>(r - 1, c - 1); _01 = img.at<float>(r - 1, c); _02 = img.at<float>(r - 1, c + 1);
                    _10 = img.at<float>(r    , c - 1);                                _12 = img.at<float>(r    , c + 1);
                    _20 = img.at<float>(r + 1, c - 1); _21 = img.at<float>(r + 1, c); _22 = img.at<float>(r + 1, c + 1);
                    
                    bool valid = false;

                    if(val > 0)
                    {
                        float vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                        if (val >= vmax)
                        {
                            _00 = prev.at<float>(r - 1, c - 1); _01 = prev.at<float>(r - 1, c); _02 = prev.at<float>(r - 1, c + 1);
                            _10 = prev.at<float>(r    , c - 1);                                 _12 = prev.at<float>(r    , c + 1);
                            _20 = prev.at<float>(r + 1, c - 1); _21 = prev.at<float>(r + 1, c); _22 = prev.at<float>(r + 1, c + 1);
                            vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                            if (val >= vmax)
                            {
                                _00 = next.at<float>(r - 1, c - 1); _01 = next.at<float>(r - 1, c); _02 = next.at<float>(r - 1, c + 1);
                                _10 = next.at<float>(r    , c - 1);                                 _12 = next.at<float>(r    , c + 1);
                                _20 = next.at<float>(r + 1, c - 1); _21 = next.at<float>(r + 1, c); _22 = next.at<float>(r + 1, c + 1);
                                vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),std::max(std::max(_12,_20),std::max(_21,_22)));
                                if (val >= vmax)
                                {
                                    float _11p = prev.at<float>(r, c), _11n = next.at<float>(r, c);
                                    valid = (val >= std::max(_11p,_11n));
                                }
                            }
                        }
                    }
                    else
                    {
                        float vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::max(_12,_20),std::min(_21,_22)));
                        if (val <= vmin)
                        {
                            _00 = prev.at<float>(r - 1, c - 1); _01 = prev.at<float>(r - 1, c); _02 = prev.at<float>(r - 1, c + 1);
                            _10 = prev.at<float>(r    , c - 1);                                 _12 = prev.at<float>(r    , c + 1);
                            _20 = prev.at<float>(r + 1, c - 1); _21 = prev.at<float>(r + 1, c); _22 = prev.at<float>(r + 1, c + 1);
                            vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));
                            if (val <= vmin)
                            {
                                _00 = next.at<float>(r - 1, c - 1); _01 = next.at<float>(r - 1, c); _02 = next.at<float>(r - 1, c + 1);
                                _10 = next.at<float>(r    , c - 1);                                 _12 = next.at<float>(r    , c + 1);
                                _20 = next.at<float>(r + 1, c - 1); _21 = next.at<float>(r + 1, c); _22 = next.at<float>(r + 1, c + 1);
                                vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));
                                if (val <= vmin)
                                {
                                    float _11p = prev.at<float>(r, c), _11n = next.at<float>(r, c);
                                    valid = (val <= std::min(_11p,_11n));
                                }
                            }
                        }
                    }

                    if(valid)
                    {
                        cv::KeyPoint kpt;
                        // 这里的坐标变换到金字塔最底层了，注意也不是原始图像那一层
                        kpt.pt.x = c * (1 << o);
                        kpt.pt.y = r * (1 << o);
                        // opencv的实现还包含其他信息
                        kpt.octave = o + (i << 8);
                        kpt.size = sigma_ * powf(2.f, i / octave_scales_) * (1 << o) * 2;
                        kpt.response = std::abs(val);
                        // opencv的实现还有角度信息
                        keypoints.push_back(kpt);
                    }
                }
            }
        }
    }
}

void Sift::retainBest(std::vector<cv::KeyPoint>& keypoints, int n_points)
{
    if( n_points >= 0 && keypoints.size() > (size_t)n_points )
    {
        if (n_points==0)
        {
            keypoints.clear();
            return;
        }
        std::nth_element(keypoints.begin(), keypoints.begin() + n_points - 1, keypoints.end(), KeypointResponseGreater());
        float ambiguous_response = keypoints[n_points - 1].response;
        std::vector<cv::KeyPoint>::const_iterator new_end = std::partition(keypoints.begin() + n_points, keypoints.end(), KeypointResponseGreaterThanOrEqualToThreshold(ambiguous_response));
        keypoints.resize(new_end - keypoints.begin());
    }
}

void Sift::resumeScale(std::vector<cv::KeyPoint>& keypoints)
{
    for( size_t i = 0; i < keypoints.size(); i++ )
    {
        cv::KeyPoint& kpt = keypoints[i];
        float scale = 0.5;
        // kpt.octave = (kpt.octave & ~255) | ((kpt.octave -1) & 255);
        kpt.pt *= scale;
        kpt.size *= scale;
    }
}

void Sift::calcDescriptor(const std::vector<cv::Mat>& gaussian_pyr, const std::vector<cv::KeyPoint>& kps, std::vector<std::vector<float>>& desc)
{
    // 计算所有图像像素x方向梯度，y方向梯度
    // 计算每个像素的梯度norm，梯度方向
    // 特征点周围取patch，patch的norm高斯滤波一下
    // patch分成4x4的cell，每个cell是8个bin，bin的值是累加的norm（加权过）
    // 连接成128向量，归一化使norm=1

    desc.clear();
    desc.resize(kps.size());

    std::vector<cv::Mat> pyr_dx(octaves_);
    std::vector<cv::Mat> pyr_dy(octaves_);
    std::vector<cv::Mat> pyr_dnorm(octaves_);

    for(int i = 0; i < octaves_; i++)
    {
        cv::Sobel(gaussian_pyr[i], pyr_dx[i], CV_32F, 1, 0, 3);
        cv::Sobel(gaussian_pyr[i], pyr_dy[i], CV_32F, 0, 1, 3);

        pyr_dnorm[i] = cv::Mat::zeros(gaussian_pyr[i].size(), CV_32FC1);
        for(int r = 0; r < gaussian_pyr[i].rows; r++)
        {
            for(int c = 0; c < gaussian_pyr[i].cols; c++)
            {
                pyr_dnorm[i].at<float>(r, c) = std::sqrt(pyr_dx[i].at<float>(r, c) * pyr_dx[i].at<float>(r, c) + pyr_dy[i].at<float>(r, c) * pyr_dy[i].at<float>(r, c));
            }
        }
    }

    for(int i = 0; i < (int)kps.size(); i++)
    {
        const cv::KeyPoint& kpt = kps[i];
        int o = kpt.octave & 255;
        int layer = (kpt.octave >> 8) & 255;
        const cv::Mat& img = gaussian_pyr[o * (octave_scales_ + 3) + layer];
    }
}

}