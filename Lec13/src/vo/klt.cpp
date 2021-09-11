#include "klt.h"

#include "util/image_util.h"
#include "util/math_util.h"

namespace mango
{
KanadeLucasTomasi::KanadeLucasTomasi() {}
KanadeLucasTomasi::~KanadeLucasTomasi() {}

/**
 * 在第2帧图像中跟踪第一帧图像点(x,y)，通过取patch的方式，迭代优化仿射变换矩阵参数，得到最终的仿射变换参数
*/
Eigen::Matrix<float, 2, 3> KanadeLucasTomasi::trackPointKLT(const cv::Mat& img_ref, const cv::Mat& img_query, float x, float y, float patch_radius, int num_iters)
{
    Eigen::Matrix<float, 2, 3> warp_transform = getWarp(0, 0, 0, 1);
    // 参考帧取patch
    cv::Mat patch_ref = getWarpedPatch(img_ref, warp_transform, x, y, patch_radius);

    int n = patch_radius * 2 + 1;
    cv::Mat xs = cv::Mat::zeros(cv::Size(1, n), CV_32F);
    cv::Mat ys = cv::Mat::zeros(cv::Size(1, n), CV_32F);
    for(int i = 0; i < n; i++)
    {
        xs.at<float>(i, 0) = float(i - patch_radius);
        ys.at<float>(i, 0) = float(i - patch_radius);
    }
    cv::Mat xy1;
    cv::hconcat(std::vector<cv::Mat>{kronecker(xs, cv::Mat::ones(cv::Size(1, n), CV_32F)), kronecker(cv::Mat::ones(cv::Size(1, n), CV_32F), ys), cv::Mat::ones(cv::Size(1, n * n), CV_32F)}, xy1);
    cv::Mat dwdx = kronecker(xy1, cv::Mat::eye(cv::Size(2,2), CV_32F));
    // std::cout << dwdx << std::endl;
    for(int i = 0; i < num_iters; i++)
    {
        // 查询帧取patch
        cv::Mat patch_query_big = getWarpedPatch(img_query, warp_transform, x, y, patch_radius + 1);
        cv::Mat patch_query = patch_query_big.rowRange(1, patch_query_big.rows - 1).colRange(1, patch_query_big.cols - 1);

        // 查询帧patch计算梯度
        cv::Mat dx = cv::Mat::zeros(patch_query.size(), CV_32F);
        cv::Mat dy = cv::Mat::zeros(patch_query.size(), CV_32F);

        for(int r = 0; r < patch_query_big.rows; r++)
        {
            for(int c = 0; c < patch_query_big.cols; c++)
            {
                if(r >= 1 && c >= 1 && r < patch_query_big.rows - 1 && c < patch_query_big.cols - 1)
                {
                    dx.at<float>(r - 1, c - 1) = float(patch_query_big.at<uchar>(r, c + 1) - patch_query_big.at<uchar>(r, c - 1));
                    dy.at<float>(r - 1, c - 1) = float(patch_query_big.at<uchar>(r + 1, c) - patch_query_big.at<uchar>(r - 1, c));
                }
            }
        }

        // cv::Mat kernel = (cv::Mat_<uchar>(1,3) << -1, 0, 1);
        // cv::filter2D(patch_query, dx, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);
        // cv::filter2D(patch_query, dy, CV_32F, kernel.t(), cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);

        // cv::imshow("1", dx);
        // cv::imshow("2", dy);
        // cv::waitKey(-1);
        // cv::destroyAllWindows();
        
        // cv::Sobel(patch_query, dx, CV_32F, 1, 0, 3);
        // cv::Sobel(patch_query, dy, CV_32F, 0, 1, 3);

        cv::Mat dx_vec = cv::Mat::zeros(cv::Size(1, n * n), CV_32F);
        cv::Mat dy_vec = cv::Mat::zeros(cv::Size(1, n * n), CV_32F);
        int cnt = 0;
        for(int r = 0; r < n; r++)
        {
            for(int c = 0; c < n; c++)
            {
                dx_vec.at<float>(cnt, 0) = dx.at<float>(c, r);
                dy_vec.at<float>(cnt, 0) = dy.at<float>(c, r);
                cnt += 1;
            }
        }
        
        cv::Mat didw;
        cv::hconcat(dx_vec, dy_vec, didw);
        cv::Mat didp = cv::Mat::zeros(cv::Size(6, n * n), CV_32F);
        for(int j = 0; j < n * n; j++)
        {
            didp.row(j) = didw.row(j) * dwdx.rowRange(j*2, j*2+2);
        }

        cv::Mat H = didp.t() * didp;
        // std::cout << H << std::endl;
        cv::Mat diff = patch_ref - patch_query;
        cv::Mat diff_vec = cv::Mat::zeros(cv::Size(1, n * n), CV_32F);
        cnt = 0;
        for(int r = 0; r < n; r++)
        {
            for(int c = 0; c < n; c++)
            {
                diff_vec.at<float>(cnt, 0) = float(diff.at<uchar>(c, r));
                cnt += 1;
            }
        }
        // std::cout << diff_vec << std::endl;
        cv::Mat delta_p = H.inv() * didp.t() * diff_vec;
        // std::cout << delta_p << std::endl;

        cnt = 0;
        for(int c = 0; c < 3; c++)
        {
            for(int r = 0; r < 2; r++)
            {
                warp_transform(r, c) += delta_p.at<float>(cnt, 0);
                cnt += 1;
            }
        }

        std::cout << cv::norm(delta_p) << std::endl;
        if(cv::norm(delta_p) < 1e-3)
        {
            break;
        }
    }

    return warp_transform;
}
}