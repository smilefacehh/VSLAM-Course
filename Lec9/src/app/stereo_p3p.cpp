#include <stdio.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include "util/file_util.h"
#include "util/math_util.h"

float thresh = 100;

int main(int argc, char** argv)
{
    // 加载第一张图像的关键点坐标、3D坐标
    Eigen::MatrixXf first_img_kpts = mango::load_matrix("../data/keypoints.txt", 516, 2);
    Eigen::MatrixXf first_img_landmarks = mango::load_matrix("../data/p_W_landmarks.txt", 516, 3);
    Eigen::Matrix3f intrinsic = mango::load_matrix("../data/K.txt", 3, 3);

    // 转换成cv形式
    std::vector<cv::KeyPoint> first_kpts, query_kpts;
    cv::Mat first_desc, query_desc;
    std::vector<cv::Point3f> first_landmarks;
    cv::Mat K = cv::Mat::zeros(cv::Size(3,3), CV_32FC1);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            K.at<float>(i,j) = intrinsic(i,j);
        }
    }

    for(int i = 0; i < first_img_kpts.rows(); i++)
    {
        first_kpts.push_back(cv::KeyPoint(cv::Point2f(first_img_kpts(i,1), first_img_kpts(i,0)), 1));
    }
    for(int i = 0; i < first_img_landmarks.rows(); i++)
    {
        first_landmarks.push_back(cv::Point3f(first_img_landmarks(i,0), first_img_landmarks(i,1), first_img_landmarks(i,2)));
    }

    // 读取第一帧图像，计算brief描述子
    auto brief = cv::xfeatures2d::BriefDescriptorExtractor::create();
    cv::Mat first_img = cv::imread("../data/000000.png", cv::IMREAD_GRAYSCALE);
    brief->compute(first_img, first_kpts, first_desc);

    // {
    //     cv::Mat harris_resp = cv::Mat::zeros(first_img.size(), CV_32FC1);
    //     cv::cornerHarris(first_img, harris_resp, 9, 3, 0.08);
    //     cv::Mat harris_resp_norm;
    //     cv::normalize(harris_resp, harris_resp_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    //     for(int r = 0; r < harris_resp_norm.rows; r++)
    //     {
    //         for(int c = 0; c < harris_resp_norm.cols; c++)
    //         {
    //             if((int) harris_resp_norm.at<float>(r,c) > thresh)
    //             {
    //                 first_kpts.push_back(cv::KeyPoint(cv::Point2f(c,r), 1));
    //             }
    //         }
    //     }
    //     brief->compute(first_img, first_kpts, first_desc);
    // }

    for(int i = 1; i <= 9; i++)
    {
        query_kpts.clear();

        // 读取图像
        char _t[100];
        sprintf(_t, "../data/%06d.png", i);
        cv::Mat query_img = cv::imread(std::string(_t), cv::IMREAD_GRAYSCALE);

        // 提取harris角点、计算描述子
        cv::Mat harris_resp = cv::Mat::zeros(query_img.size(), CV_32FC1);
        cv::cornerHarris(query_img, harris_resp, 9, 3, 0.08);
        cv::Mat harris_resp_norm;
        cv::normalize(harris_resp, harris_resp_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        for(int r = 0; r < harris_resp_norm.rows; r++)
        {
            for(int c = 0; c < harris_resp_norm.cols; c++)
            {
                if((int) harris_resp_norm.at<float>(r,c) > thresh)
                {
                    query_kpts.push_back(cv::KeyPoint(cv::Point2f(c,r), 1));
                }
            }
        }
        brief->compute(query_img, query_kpts, query_desc);

        // 匹配
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches;
        matcher.match(first_desc, query_desc, matches);

        // 过滤匹配点
        int match_N = matches.size();
        std::vector<cv::Point3f> match_landmarks;
        std::vector<cv::Point2f> match_query_pts;

        for(int j = 0; j < matches.size(); j++)
        {
            match_landmarks.push_back(first_landmarks[matches[j].queryIdx]);
            match_query_pts.push_back(query_kpts[matches[j].trainIdx].pt);
        }

        // 迭代ransac p3p
        float px_err = 100; // 重投影像素误差10以内，认为是inlier
        int max_num_inliers = 0;
        cv::Mat best_R, best_t;
        int num_iterations = 100; // 迭代次数
        std::vector<int> match_final;
        for(int k = 0; k < num_iterations; k++)
        {
            // 随机3对点
            std::vector<int> rnd_index = mango::randomN(match_N, 3);
            std::vector<cv::Point3f> sample_landmarks;
            std::vector<cv::Point2f> sample_query_pts;
            for(int j = 0; j < 3; j++)
            {
                sample_landmarks.push_back(match_landmarks[rnd_index[j]]);
                sample_query_pts.push_back(match_query_pts[rnd_index[j]]);
            }

            // p3p求解R、t
            std::vector<cv::Mat> rvecs, tvecs;
            cv::Mat R,t;
            cv::solveP3P(sample_landmarks, sample_query_pts, K, cv::Mat(), rvecs, tvecs, cv::SOLVEPNP_AP3P);
            R = rvecs[0], t = tvecs[0];

            std::vector<int> match_filter;
            std::vector<cv::Point2f> projected_query_pts;
            cv::projectPoints(match_landmarks, R, t, K, cv::Mat(), projected_query_pts);
            int inliers = 0;
            for(int j = 0; j < match_N; j++)
            {
                float dx = match_query_pts[j].x - projected_query_pts[j].x;
                float dy = match_query_pts[j].y - projected_query_pts[j].y;
                if(dx*dx + dy*dy < px_err * px_err)
                {
                    inliers++;
                    match_filter.push_back(j);
                }
            }
            if(inliers > max_num_inliers)
            {
                max_num_inliers = inliers;
                best_R = R;
                best_t = t;
                match_final = match_filter;
            }
        }
        std::cout << "--------------------" << std::string(_t) << "---------------------" << std::endl;
        std::cout << "total match:" << match_N << ", inlier match:" << max_num_inliers << std::endl;
        std::cout << "R:\n" << best_R << std::endl;
        std::cout << "t:\n" << best_t << std::endl;

        std::vector<cv::DMatch> match_ransac;
        for(int j = 0; j < match_final.size(); j++)
        {
            match_ransac.push_back(matches[j]);
        }
        cv::Mat img_match;
        cv::drawMatches(first_img, first_kpts, query_img, query_kpts, match_ransac, img_match);
        cv::imshow("1", img_match);
        cv::waitKey(-1);
        cv::destroyAllWindows();
    }
    return 0;
}