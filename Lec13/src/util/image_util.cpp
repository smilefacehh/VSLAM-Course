#include "image_util.h"

namespace mango
{

/**
 * 合并图像
 * @param imgs 图像
 * @param w    图像缩放为指定大小，宽
 * @param h    图像缩放为指定大小，高
*/
cv::Mat mergeImage(const std::vector<cv::Mat>& imgs, int w, int h)
{
    int n = (int)imgs.size();
    int merge_w, merge_h;
    if(n <= 3)
    {
        merge_w = w * n;
        merge_h = h;
    }
    else if(n % 2 == 0)
    {
        merge_w = w * n / 2;
        merge_h = h * 2;
    }
    else
    {
        merge_w = w * n / 2 + 1;
        merge_h = h * 2;
    }

    cv::Mat merge_img(merge_h, merge_w, CV_8UC3);
    for(int i = 0; i < n; i++)
    {
        int row, col;
        if(n <= 3)
        {
            row = 0;
            col = i;
        }
        else if(n % 2 == 0)
        {
            row = i < n / 2 ? 0 : 1;
            col = i % (n / 2);
        }
        else
        {
            row = i < (n / 2 + 1) ? 0 : 1;
            col = i % (n / 2 + 1);
        }

        cv::Mat tmp;
        cv::resize(imgs[i], tmp, cv::Size(w, h));

        cv::Mat img_colorful(h, w, CV_8UC3);
        if(tmp.channels() == 1)
        {
            std::vector<cv::Mat> channels(3, tmp);
            cv::merge(channels, img_colorful);
        }
        else
        {
            tmp.copyTo(img_colorful);
        }

        img_colorful.copyTo(merge_img(cv::Rect(col * w, row * h, w, h)));
    }
    return merge_img;
}

cv::Mat gray2color(const cv::Mat& img)
{
    double vmin, vmax;
    cv::minMaxLoc(img, &vmin, &vmax);
    double alpha = (255.0 / (vmax - vmin)) * 1;
    cv::Mat tmp;
    img.convertTo(tmp, CV_8U, alpha, -vmin * alpha);
    cv::Mat color;
    applyColorMap(tmp, color, cv::COLORMAP_JET);
    return color;
}

/**
 * 图像仿射变换，旋转、平移、缩放
*/
cv::Mat warpImage(const cv::Mat& img, const Eigen::Matrix<float, 2, 3>& warpTransform)
{
    assert(img.channels() == 1);

    cv::Mat warp_img = cv::Mat::zeros(img.size(), img.type());
    for(int r = 0; r < img.rows; r++)
    {
        for(int c = 0; c < img.cols; c++)
        {
            Eigen::Vector3f coord(c, r, 1);
            Eigen::Vector2f new_coord = warpTransform * coord;
            int new_r = new_coord(1), new_c = new_coord(0);
            if(new_c >= 0 && new_c < img.cols && new_r >= 0 && new_r < img.rows)
            {
                warp_img.at<uchar>(new_r, new_c) = img.at<uchar>(r, c);
            }
        }
    }
    
    return warp_img;
}

/**
 * (2,3)仿射变换矩阵，近似
*/
Eigen::Matrix<float, 2, 3> getWarp(float dx, float dy, float alpha, float lambda)
{
    Eigen::Matrix<float, 2, 3> warpTransform;
    float c = cosf(alpha), s = sinf(alpha);
    warpTransform << c, -s, dx,
                     s,  c, dy;
    warpTransform *= lambda;

    return warpTransform;
}

/**
 * 取点（x,y）处仿射变换后的patch
*/
cv::Mat getWarpedPatch(const cv::Mat& img, const Eigen::Matrix<float, 2, 3>& warpTransform, float x, float y, float patch_radius)
{
    assert(img.channels() == 1);

    cv::Mat patch = cv::Mat::zeros(cv::Size(2*patch_radius+1, 2*patch_radius+1), CV_8UC1);

    for(int i = -patch_radius; i <= patch_radius; i++)
    {
        for(int j = -patch_radius; j <= patch_radius; j++)
        {
            Eigen::Vector2f warped = Eigen::Vector2f(x, y) + warpTransform * Eigen::Vector3f(i, j, 1);
            float warp_x = warped(0); float warp_y = warped(1);
            if(warp_x >= 0 && warp_x < img.cols && warp_y >= 0 && warp_y < img.rows)
            {
                patch.at<uchar>(j + patch_radius, i + patch_radius) = img.at<uchar>(int(warp_y), int(warp_x));
            }
        }
    }

    return patch;
}

}