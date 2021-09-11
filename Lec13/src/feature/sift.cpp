#include "sift.h"

#include <assert.h>

#include "util/draw_util.h"
#include "util/image_util.h"
#include "util/math_util.h"

namespace mango
{
Sift::Sift(int nfeatures, int octaves, int octave_scales, double contrast_threshold, double edge_threshold, double sigma)
    : nfeatures_(nfeatures), octaves_(octaves), octave_scales_(octave_scales), contrast_threshold_(contrast_threshold), edge_threshold_(edge_threshold), sigma_(sigma)
{
    std::cout << "contrast_threshold_:" << contrast_threshold_ << std::endl;
    // sift_ = cv::SIFT::create(100);
}

Sift::~Sift() {}

void Sift::detect(const cv::Mat& src, std::vector<cv::KeyPoint>& keypoints, std::vector<std::vector<float>>& descriptors)
{
    cv::Mat base = createInitialImage(src);
    buildGaussianPyramid(base, gaussian_pyr_);
    buildDogPyramid(gaussian_pyr_, dog_pyr_);

    findScaleSpaceExtrema(dog_pyr_, keypoints);
    std::cout << "findScaleSpaceExtrema:" << keypoints.size() << std::endl;
    retainBest(keypoints, nfeatures_);
    std::cout << "retainBest:" << keypoints.size() << std::endl;
    resumeScale(keypoints);
    std::cout << "resumeScale:" << keypoints.size() << std::endl;

    calcDescriptor(gaussian_pyr_, keypoints, descriptors);
}

void Sift::match(const std::vector<std::vector<float>>& reference_desc, const std::vector<std::vector<float>>& query_desc, std::vector<int>& match_)
{
    // 描述子匹配
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
            double ssd = mango::ssd<float>(query_desc[i], reference_desc[j]);
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

    global_min_ssd *= 1.1;

    for(int i = 0; i < num_kp; i++)
    {
        if(ssd_vec[i] >= global_min_ssd)
        {
            match_[i] = -1;
        }
    }
}

void Sift::plotMatchTwoImage(const cv::Mat& reference_img, const cv::Mat& query_img, const std::vector<cv::KeyPoint>& reference_kps, const std::vector<cv::KeyPoint>& query_kps, const std::vector<int>& match, const std::string& saved_path, bool saved)
{
    cv::Mat img_kp_ref = mango::drawKeyPoint(reference_img, reference_kps, cv::Scalar(0, 0, 255));
    cv::Mat img_kp_query = mango::drawKeyPoint(query_img, query_kps, cv::Scalar(0, 0, 255));
    cv::Mat merge_img = mango::mergeImage(std::vector<cv::Mat>{img_kp_ref, img_kp_query}, img_kp_query.cols, img_kp_query.rows);
    for(int i = 0; i < (int)match.size(); i++)
    {
        if(match[i] != -1)
        {
            cv::Point2f pt_ref, pt_query;
            pt_ref.x = reference_kps[match[i]].pt.x;
            pt_ref.y = reference_kps[match[i]].pt.y;
            pt_query.x = query_kps[i].pt.x + reference_img.cols;
            pt_query.y = query_kps[i].pt.y;
            cv::line(merge_img, pt_ref, pt_query, cv::Scalar(0, 255, 0));
        }
    }
    if(saved)
    {
        cv::imwrite(saved_path, merge_img);
    }
    cv::imshow("match", merge_img);
    cv::waitKey(-1);
    cv::destroyWindow("match");
}

void Sift::plotGaussianPyramid(const std::string& saved_path, bool saved)
{
    if(gaussian_pyr_.empty())
    {
        return;
    }
    
    int w = gaussian_pyr_[0].cols, h = gaussian_pyr_[0].rows;
    int n = gaussian_pyr_.size() / octaves_;
    plotPyramid(gaussian_pyr_, w / 5, h / 5, n, saved_path, saved);
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
        // dog_pyr_norm[i] = cv::Mat::zeros(dog_pyr_[i].size(), CV_32FC1);
        // for(int r = 0; r < dog_pyr_[i].rows; r++)
        // {
        //     for(int c = 0; c < dog_pyr_[i].cols; c++)
        //     {
        //         if(dog_pyr_[i].at<float>(r, c) < 0)
        //         {
        //             dog_pyr_norm[i].at<float>(r, c) = -1 * dog_pyr_[i].at<float>(r, c);
        //         }
        //         else
        //         {
        //             dog_pyr_norm[i].at<float>(r, c) = dog_pyr_[i].at<float>(r, c);
        //         }
        //     }
        // }
        cv::normalize(dog_pyr_[i], dog_pyr_norm[i], 0, 255, cv::NORM_MINMAX);
    }
    int w = dog_pyr_norm[0].cols, h = dog_pyr_norm[0].rows;
    int n = dog_pyr_norm.size() / octaves_;
    plotPyramid(dog_pyr_norm, w / 5, h / 5, n, saved_path, saved);
}

void Sift::plotPyramid(const std::vector<cv::Mat>& pyr, int width, int height, int n, const std::string& saved_path, bool saved)
{
    int octave = pyr.size() / n;
    cv::Mat pyr_img = cv::Mat::zeros((2 - std::pow(2, -(octave - 1))) * height, n * width, CV_8UC1);

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

void Sift::plotKeypoints(const cv::Mat& src, const std::vector<cv::KeyPoint>& kps, const std::string& saved_path, bool saved)
{
    std::cout << kps.size() << std::endl;
    for(size_t i = 0; i < kps.size(); i++)
    {
        std::cout << kps[i].pt.x << "," << kps[i].pt.y << "," << kps[i].size << std::endl;
    }
    cv::Mat img_kp = mango::drawKeyPoint(src, kps, cv::Scalar(0, 0, 255));
    // cv::resize(img_kp, img_kp, cv::Size(img_kp.cols, img_kp.rows));
    if(saved)
    {
        cv::imwrite(saved_path, img_kp);
    }
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
    std::cout << "sig_diff:" << sig_diff << std::endl;
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
        cv::subtract(src2, src1, dst, cv::noArray());
        cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
        // std::cout << dst << std::endl;
        // cv::imshow("a", dst);
        // cv::waitKey(-1);
        // cv::destroyWindow("a");
    }
}

void Sift::findScaleSpaceExtrema(const std::vector<cv::Mat>& dog_pyr, std::vector<cv::KeyPoint>& keypoints)
{
    // 边界不提取特征点
    int border = 8;
    for(int o = 0; o < octaves_; o++)
    {
        for(int i = 1; i <= octave_scales_; i++)
        {
            int idx = o * (octave_scales_ + 2) + i;
            const cv::Mat& img = dog_pyr[idx];
            const cv::Mat& prev = dog_pyr[idx - 1];
            const cv::Mat& next = dog_pyr[idx + 1];

            // cv::Mat merge = mango::mergeImage(std::vector<cv::Mat>{prev, img, next}, img.cols/3, img.rows/3);
            // cv::Mat merge_norm;
            // cv::normalize(merge, merge_norm, 0, 255, cv::NORM_MINMAX);
            // cv::imshow("merge", merge_norm);
            // cv::waitKey(-1);
            // cv::destroyWindow("merge");

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
                        float vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),std::min(std::min(_12,_20),std::min(_21,_22)));
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
    // 1.计算所有图像像素x方向梯度，y方向梯度
    // 2.计算每个像素的梯度norm，梯度方向
    // 3.特征点周围取patch，patch的norm高斯滤波一下
    // 4.patch分成4x4的cell，每个cell是8个bin，bin的值是累加的norm（加权过）
    // 5.连接成128向量，归一化使norm=1

    desc.clear();
    desc.resize(kps.size());

    int n = (int)gaussian_pyr.size();

    std::vector<cv::Mat> pyr_dx(n);
    std::vector<cv::Mat> pyr_dy(n);
    std::vector<cv::Mat> pyr_dnorm(n);
    std::vector<cv::Mat> pyr_dir(n);
    
    // 1,2
    for(int i = 0; i < n; i++)
    {
        cv::Sobel(gaussian_pyr[i], pyr_dx[i], CV_32F, 1, 0, 3);
        cv::Sobel(gaussian_pyr[i], pyr_dy[i], CV_32F, 0, 1, 3);

        pyr_dnorm[i] = cv::Mat::zeros(gaussian_pyr[i].size(), CV_32FC1);
        pyr_dir[i] = cv::Mat::zeros(gaussian_pyr[i].size(), CV_32FC1);

        for(int r = 0; r < gaussian_pyr[i].rows; r++)
        {
            for(int c = 0; c < gaussian_pyr[i].cols; c++)
            {
                pyr_dnorm[i].at<float>(r, c) = std::sqrt(pyr_dx[i].at<float>(r, c) * pyr_dx[i].at<float>(r, c) + pyr_dy[i].at<float>(r, c) * pyr_dy[i].at<float>(r, c));

                if(pyr_dy[i].at<float>(r, c) == 0 && pyr_dx[i].at<float>(r, c) == 0)
                {
                    pyr_dir[i].at<float>(r, c) = std::numeric_limits<float>::max();
                }
                else
                {
                    pyr_dir[i].at<float>(r, c) = atan2(pyr_dy[i].at<float>(r, c), pyr_dx[i].at<float>(r, c));
                }
            }
        }
    }

    std::cout << "#1" << std::endl;
    // 3,4,5
    for(int i = 0; i < (int)kps.size(); i++)
    {
        const cv::KeyPoint& kpt = kps[i];
        int o = kpt.octave & 255; // octave
        int layer = (kpt.octave >> 8) & 255; // 在某一octave下，所在的层数
        std::cout << "o:" << o << ",layer:" << layer << std::endl;

        const cv::Mat& img_dnorm = pyr_dnorm[o * (octave_scales_ + 3) + layer];
        const cv::Mat& img_dir = pyr_dir[o * (octave_scales_ + 3) + layer];

        std::cout << "#1.1" << std::endl;

        // 特征点周围取patch
        int r, c;
        if(o == 0)
        {
            r = kpt.pt.y * 2;
            c = kpt.pt.x * 2;
        }
        else
        {
            r = kpt.pt.y / (1 << (o - 1));
            c = kpt.pt.x / (1 << (o - 1));
        }
        std::cout << "img size:" << img_dnorm.size() << std::endl;
        std::cout << "kp:" << kpt.pt << std::endl;

        cv::Mat patch_norm = img_dnorm.rowRange(r - 7, r + 9).colRange(c - 7, c + 9).clone();
        cv::Mat patch_dir = img_dir.rowRange(r - 7, r + 9).colRange(c - 7, c + 9).clone();
        std::vector<float> kp_desc;
        std::cout << "#1.2" << std::endl;
        calcPatchDescriptor(patch_norm, patch_dir, kp_desc);
        desc[i] = kp_desc;
    }
}

void Sift::calcPatchDescriptor(const cv::Mat& patch_norm, const cv::Mat& patch_dir, std::vector<float>& desc)
{
    std::cout << "#2" << std::endl;
    assert(patch_norm.rows == 16 && patch_norm.cols == 16);
    assert(patch_dir.rows == 16 && patch_dir.cols == 16);
    std::cout << "#3" << std::endl;

    desc.clear();
    desc.reserve(128);

    // patch每个梯度值首先按距离加权一下
    cv::Mat patch_norm_gaussian;
    cv::GaussianBlur(patch_norm, patch_norm_gaussian, cv::Size(), 1.5*16, 1.5*16);

    // 连接16个cell，得到128个值
    for(int i = 0; i < 16; i++)
    {
        cv::Mat cell_norm = patch_norm_gaussian.rowRange((i/4)*4, (i/4)*4 + 4).colRange((i%4)*4, (i%4)*4 + 4).clone();
        cv::Mat cell_dir = patch_dir.rowRange((i/4)*4, (i/4)*4 + 4).colRange((i%4)*4, (i%4)*4 + 4).clone();

        std::vector<float> hog;
        calcCellHoG(cell_norm, cell_dir, hog);
        desc.insert(desc.end(), hog.begin(), hog.end());
    }

    // 归一化，norm = 1
    assert(desc.size() == 128);
    float sum_square = 0.0;
    for(int i = 0; i < desc.size(); i++)
    {
        sum_square += (desc[i] * desc[i]);
    }

    if(sum_square != 0)
    {
        for(int i = 0; i < desc.size(); i++)
        {
            desc[i] /= sum_square;
        }
    }
}

void Sift::calcCellHoG(const cv::Mat& cell_norm, const cv::Mat& cell_dir, std::vector<float>& hog)
{
    assert(cell_norm.rows == 4 && cell_norm.cols == 4);
    assert(cell_dir.rows == 4 && cell_dir.cols == 4);

    hog.clear();
    hog.resize(8);

    for(int i = 0; i < 16; i++)
    {
        int r = i / 4, c = i % 4;
        if(cell_dir.at<float>(r, c) >= std::numeric_limits<float>::max()) continue;

        if(cell_dir.at<float>(r, c) >= 0)
        {
            hog[cell_dir.at<float>(r, c) / (M_PI / 4)] += cell_norm.at<float>(r, c);
        }
        else
        {
            hog[cell_dir.at<float>(r, c) / (M_PI / 4) + 7] += cell_norm.at<float>(r, c);
        }
    }
}

}