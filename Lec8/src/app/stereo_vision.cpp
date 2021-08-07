#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>

#include <random>

// opencv
#include <opencv2/opencv.hpp>

#include "vo/stereo.h"
#include "util/string_util.h"
#include "util/math_util.h"

namespace mango {

class StereoVision
{
public:
    StereoVision(const std::string& data_folder)
        : data_folder_(data_folder)
    {
        stereo_ = std::shared_ptr<mango::Stereo>(new mango::Stereo());
    }
    ~StereoVision() {}

    void run();

private:
    /**
     * 测试线性三角化函数
    */
    void test_linearTriangulation();

    /**
     * 测试八点法求基础矩阵的函数，不带噪声、带噪声的数据，归一化的方法
    */
    void test_fundamentalEightPoint();

    // 成员
    std::string data_folder_;
    std::shared_ptr<mango::Stereo> stereo_;
};

void StereoVision::run()
{    
    // 1.测试线性三角化函数
    // test_linearTriangulation();

    // 2.测试八点法计算基础矩阵，归一化点坐标，误差计算等
    // test_fundamentalEightPoint();

    // 读取两幅图像的匹配像素点，估计位姿
    Eigen::Matrix3Xf p1, p2;

    // 点一
    std::ifstream match_ifs1((data_folder_ + "matches0001.txt").c_str());
    if(!match_ifs1.is_open())
    {
        std::cout << "打开文件data/matches0001.txt失败" << std::endl;
        return;
    }
    std::istream_iterator<float> begin1(match_ifs1);
    std::istream_iterator<float> end1;
    std::vector<float> data1(begin1, end1);
    int pt_N = data1.size() / 2;
    p1.resize(3, pt_N);
    for(int i = 0; i < pt_N; i++)
    {
        p1(0, i) = data1[i];
        p1(1, i) = data1[i + pt_N];
        p1(2, i) = 1;
    }

    // 点二
    std::ifstream match_ifs2((data_folder_ + "matches0002.txt").c_str());
    if(!match_ifs2.is_open())
    {
        std::cout << "打开文件data/matches0002.txt失败" << std::endl;
        return;
    }
    std::istream_iterator<float> begin2(match_ifs2);
    std::istream_iterator<float> end2;
    std::vector<float> data2(begin2, end2);
    pt_N = data2.size() / 2;
    p2.resize(3, pt_N);
    for(int i = 0; i < pt_N; i++)
    {
        p2(0, i) = data2[i];
        p2(1, i) = data2[i + pt_N];
        p2(2, i) = 1;
    }
    // 相机内参
    Eigen::Matrix3f K;
    K << 1379.74, 0, 760.35,
         0, 1382.08, 503.41,
         0,       0,      1;

    // 计算本质矩阵
    Eigen::Matrix3f E = stereo_->estimateEssentialMatrix(p1, p2, K, K);

    // 分解E得到候选R|t
    Eigen::Matrix<float, 6, 3> rotation;
    Eigen::Vector3f u3;
    stereo_->decomposeEssentialMatrix(E, rotation, u3);

    // 通过三角化判断z值正负选择正确的一个解
    Eigen::MatrixXf T_21 = stereo_->disambiguateRelativePose(rotation, u3, p1, p2, K, K);

    // 三角化
    Eigen::MatrixXf M1 = K * Eigen::Matrix<float, 3, 4>::Identity();
    Eigen::MatrixXf M2 = K * T_21;
    Eigen::MatrixXf P = stereo_->linearTriangulation(p1, p2, M1, M2);

    // 打印
    std::cout << "本质矩阵E:" << std::endl;
    std::cout << E << std::endl;
    std::cout << "位姿T_21:" << std::endl;
    std::cout << T_21 << std::endl;
    std::cout << "3D坐标(10个):" << std::endl;
    std::cout << P.block(0,0,4,20) << std::endl;
}

void StereoVision::test_linearTriangulation()
{
    // 随机构造3D点
    int N = 10;
    Eigen::Matrix<float, 4, Eigen::Dynamic> P;
    P.setRandom(4, N);
    for(int i = 0; i < N; i++)
    {
        P.row(2)(i) = P.row(2)(i) * 5 + 10;
    }
    P.row(3).setOnes();

    // 构造投影矩阵
    Eigen::Matrix<float, 3, 4> M1, M2;
    M1 << 500, 0, 320, 0,
          0, 500, 240, 0,
          0,   0,   1, 0;
    M2 << 500, 0, 320, -100,
          0, 500, 240, 0,
          0,   0,   1, 0;

    // 计算投影点
    Eigen::MatrixXf p1 = M1 * P;
    Eigen::MatrixXf p2 = M2 * P;

    Eigen::MatrixXf P_est = stereo_->linearTriangulation(p1, p2, M1, M2);

    // 误差
    std::cout << P_est - P << std::endl;
}

void StereoVision::test_fundamentalEightPoint()
{
    // 随机构造3D点
    int N = 40;
    Eigen::Matrix<float, 4, Eigen::Dynamic> P;
    P.setRandom(4, N);
    for(int i = 0; i < N; i++)
    {
        P.row(2)(i) = P.row(2)(i) * 5 + 10;
    }
    P.row(3).setOnes();

    // 构造投影矩阵
    Eigen::Matrix<float, 3, 4> M1, M2;
    M1 << 500, 0, 320, 0,
          0, 500, 240, 0,
          0,   0,   1, 0;
    M2 << 500, 0, 320, -100,
          0, 500, 240, 0,
          0,   0,   1, 0;

    // 计算投影点
    Eigen::MatrixXf p1 = M1 * P;
    Eigen::MatrixXf p2 = M2 * P;

    // 添加噪声
    float sigma = 0.1;
    Eigen::MatrixXf tmp;
    tmp.setRandom(p1.rows(), p1.cols());
    Eigen::MatrixXf noisy_p1 = p1 + sigma * tmp;
    tmp.setRandom(p2.rows(), p2.cols());
    Eigen::MatrixXf noisy_p2 = p2 + sigma * tmp;

    // 估计基础矩阵，计算误差
    Eigen::Matrix3f F = stereo_->fundamentalEightPoint(p1, p2);
    double cost_algebraic = 0;
    for(int i = 0; i < p1.cols(); i++)
    {
        cost_algebraic += pow(p2.col(i).transpose() * F * p1.col(i), 2);
    }
    cost_algebraic = sqrt(cost_algebraic / p1.cols());
    double cost_dist_epiline = stereo_->distPoint2EpipolarLine(F, p1, p2);

    std::cout << "----无噪声的八点法估计基础矩阵----" << std::endl;
    std::cout << "代数误差：" << cost_algebraic << std::endl;
    std::cout << "几何误差：" << cost_dist_epiline << std::endl;

    // 带噪声估计，计算误差
    F = stereo_->fundamentalEightPoint(noisy_p1, noisy_p2);
    cost_algebraic = 0;
    for(int i = 0; i < noisy_p1.cols(); i++)
    {
        cost_algebraic += pow(noisy_p2.col(i).transpose() * F * noisy_p1.col(i), 2);
    }
    cost_algebraic = sqrt(cost_algebraic / noisy_p1.cols());
    cost_dist_epiline = stereo_->distPoint2EpipolarLine(F, noisy_p1, noisy_p2);

    std::cout << "----带噪声的八点法估计基础矩阵----" << std::endl;
    std::cout << "代数误差：" << cost_algebraic << std::endl;
    std::cout << "几何误差：" << cost_dist_epiline << std::endl;

    // 归一化八点法，带噪声
    // noisy_p1 = p1;
    // noisy_p2 = p2;
    F = stereo_->fundamentalEightPointNormalized(noisy_p1, noisy_p2);
    cost_algebraic = 0;
    for(int i = 0; i < noisy_p1.cols(); i++)
    {
        cost_algebraic += pow(noisy_p2.col(i).transpose() * F * noisy_p1.col(i), 2);
    }
    cost_algebraic = sqrt(cost_algebraic / noisy_p1.cols());
    cost_dist_epiline = stereo_->distPoint2EpipolarLine(F, noisy_p1, noisy_p2);

    std::cout << "----带噪声的归一化八点法估计基础矩阵----" << std::endl;
    std::cout << "代数误差：" << cost_algebraic << std::endl;
    std::cout << "几何误差：" << cost_dist_epiline << std::endl;
}

}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cout << "Usage: stereo_vision ../data" << std::endl;
        return 0;
    }

    std::string data_folder = mango::folder_add_slash(argv[1]);

    mango::StereoVision stereo_vision(data_folder);
    stereo_vision.run();

    return 0;
}