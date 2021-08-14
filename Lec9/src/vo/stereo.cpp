#include "stereo.h"

#include <limits.h>
#include <omp.h>

#include "util/math_util.h"


namespace mango
{
Stereo::Stereo() {}
Stereo::~Stereo() {}

/**
 * 对齐的双目图像计算视差
 * @param left_img      左图
 * @param right_img     右图
 * @param patch_radius  半径，2r+1
 * @param min_disp      最小视差
 * @param max_disp      最大视差
*/
cv::Mat Stereo::match(const cv::Mat& left_img, const cv::Mat& right_img, float patch_radius, float min_disp, float max_disp)
{
    cv::Mat disp_img = cv::Mat::zeros(left_img.size(), CV_32FC1);
    
    int rmax = left_img.rows - (int)patch_radius, cmax = left_img.cols - (int)patch_radius;

    // 遍历左图像素点
#pragma omp parallel for
    for(int r = patch_radius; r < rmax; r++)
    {
        int rstart = r-patch_radius, rend = r+patch_radius+1;

#pragma omp parallel for
        for(int c = max_disp + patch_radius; c < cmax; c++)
        {
            int cstart = c-patch_radius, cend = c+patch_radius+1;

            // 所有候选滑窗的ssd值
            std::vector<double> ssd_vals(max_disp - min_disp + 1, 0);

            // 最小的ssd值
            double min_ssd_val = std::numeric_limits<double>::max();
            // 最小的ssd值对应的视差
            int min_ssd_disp = -1;
            
            cv::Mat left_patch = left_img.rowRange(rstart, rend).colRange(cstart, cend); // 定义放里面，放外面不行
            // 遍历右图扫描线，范围
            for(int i = min_disp; i <= max_disp; i++)
            {
                cv::Mat right_patch = right_img.rowRange(rstart, rend).colRange(cstart-i, cend-i);
                double ssd_val = mango::ssd<uchar>(left_patch, right_patch);
                ssd_vals[i - min_disp] = ssd_val;
                if(ssd_val < min_ssd_val)
                {
                    min_ssd_val = ssd_val;
                    min_ssd_disp = i;
                }
            }
            
            // 边界上的点也不考虑，因为后面要用相邻两个点拟合二次曲线，找最低点，亚像素插值
            if(min_ssd_disp == -1 || min_ssd_disp == min_disp || min_disp == max_disp)
            {
                disp_img.at<float>(r, c) = 0;
            }
            else
            {
                // 超过3个ssd比较小的点，这个点认为不够准，视差设为0
                int count = 0;
                int th = 3;
                for(int i = 0; i < max_disp - min_disp + 1; i++)
                {
                    if(ssd_vals[i] <= 1.5 * min_ssd_val)
                    {
                        count++;
                        if(count >= th)
                        {
                            break;
                        }
                    }
                }
                if(count >= th)
                {
                    disp_img.at<float>(r, c) = 0;
                }
                else
                {
                    // 3个值拟合二次曲线，求极值点
                    double min_ssd_disp_left = min_ssd_disp - 1, min_ssd_disp_right = min_ssd_disp + 1;
                    double min_ssd_left = ssd_vals[min_ssd_disp_left - min_disp], min_ssd = ssd_vals[min_ssd_disp - min_disp], min_ssd_right = ssd_vals[min_ssd_disp_right - min_disp];

                    cv::Mat K = mango::polyfit(std::vector<double>{min_ssd_disp_left,(double)min_ssd_disp,min_ssd_disp_right},
                                               std::vector<double>{min_ssd_left,min_ssd,min_ssd_right},
                                               2);

                    if(K.at<double>(2,0) <= 0 || abs(-K.at<double>(1,0)/(2 * K.at<double>(2,0)) - min_ssd_disp) >= 1)
                    {
                        disp_img.at<float>(r, c) = min_ssd_disp;
                    }
                    else
                    {
                        disp_img.at<float>(r, c) = -K.at<double>(1,0)/(2 * K.at<double>(2,0));
                    }
                }
            }
        }
    }

    return disp_img;
}

/**
 * 视差图转换成点云
 * @param disparity 视差图
 * @param img       原始左图
 * @param depth     深度图
 * @param K         相机内参
 * @param baseline  基线
*/
pcl::PointCloud<pcl::PointXYZRGB> Stereo::disparity2pointcloud(const cv::Mat& disparity, const cv::Mat& img, cv::Mat& depth, const Eigen::Matrix3d& K, double baseline)
{
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
    // 临时存
    cv::Mat postmp = cv::Mat(disparity.size(), CV_32FC3);
    depth = cv::Mat(disparity.size(), CV_32FC1);

#pragma omp parallel for
    for(int r = 0; r < disparity.rows; r++)
    {
#pragma omp parallel for
        for(int c = 0; c < disparity.cols; c++)
        {
            if(disparity.at<float>(r, c) > 0)
            {
                Eigen::Vector3d p0(r, c, 1);
                Eigen::Vector3d p1(r, c - disparity.at<float>(r,c), 1);
                Eigen::Vector3d p0_ = K.inverse() * p0;
                Eigen::Vector3d p1_ = K.inverse() * p1;
                Eigen::Matrix<double, 3, 2> A;
                A.block(0, 0, 3, 1) = p0_;
                A.block(0, 1, 3, 1) = p1_;
                Eigen::Vector3d b(baseline, 0, 0);
                Eigen::Vector2d lambda = (A.transpose() * A).inverse() * (A.transpose() * b);
                Eigen::Vector3d P = lambda(0) * K.inverse() * p0;
                depth.at<float>(r, c) = P(2);
                postmp.at<cv::Vec3f>(r, c)[0] = P(0);
                postmp.at<cv::Vec3f>(r, c)[1] = P(1);
                postmp.at<cv::Vec3f>(r, c)[2] = P(2);
            }
        }
    }

    // 并行for里面不能用push_back
    for(int r = 0; r < postmp.rows; r++)
    {
        for(int c = 0; c < postmp.cols; c++)
        {
            if(disparity.at<float>(r,c) > 0)
            {
                pcl::PointXYZRGB p;
                p.x = postmp.at<cv::Vec3f>(r, c)[0];
                p.y = postmp.at<cv::Vec3f>(r, c)[1];
                p.z = postmp.at<cv::Vec3f>(r, c)[2];
                if(img.channels() == 3)
                {
                    p.b = img.at<cv::Vec3f>(r, c)[0];
                    p.g = img.at<cv::Vec3f>(r, c)[1];
                    p.r = img.at<cv::Vec3f>(r, c)[2];
                }
                else
                {
                    p.b = img.at<uchar>(r, c);
                    p.g = p.b;
                    p.r = p.b;
                }
                pointcloud.push_back(p);
            }
        }
    }
    return pointcloud;
}

/**
 * 线性三角化
 * @param p1 (3,N) 图像一中的像素点齐次坐标
 * @param p2 (3,N) 图像二中的像素点齐次坐标
 * @param M1 (3,4) 图像一的投影矩阵，M=K[R|t]
 * @param M2 (3,4) 图像二的投影矩阵，M=K[R|t]
 * @return   (4,N) 3D点的齐次坐标，参考坐标系为图像一的坐标系
*/
Eigen::Matrix<float, 4, Eigen::Dynamic> Stereo::linearTriangulation(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1,
                                                                    const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2,
                                                                    const Eigen::Matrix<float, 3, 4>& M1,
                                                                    const Eigen::Matrix<float, 3, 4>& M2)
{
    assert(p1.size() == p2.size());

    int pt_N = p1.cols();

    Eigen::Matrix<float, 4, Eigen::Dynamic> Pt;
    Pt.resize(4, pt_N);

    for(int i = 0; i < pt_N; i++)
    {
        Eigen::Matrix<float, 6, 4> A;
        Eigen::Matrix3f p1_skew = mango::skewSymmetricMatrix3<float>(p1.col(i));
        Eigen::Matrix3f p2_skew = mango::skewSymmetricMatrix3<float>(p2.col(i));
        A.block(0, 0, 3, 4) = p1_skew * M1;
        A.block(3, 0, 3, 4) = p2_skew * M2;

        // A = USV^T，这里的V就是公式里的V
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf U, S, V;
        U = svd.matrixU();
        V = svd.matrixV();
        S = svd.singularValues();
        Eigen::Vector4f solu = V.col(V.cols() - 1);

        // 保证齐次
        solu /= solu(3);

        Pt.col(i) = solu;
    }

    return Pt;
}


/**
 * 八点法计算基础矩阵
 * 约束|F|=0，这样所有的极线才会相交于极点
 * @param p1 (3,N) 图像一中的像素点齐次坐标
 * @param p2 (3,N) 图像二中的像素点齐次坐标
 * @return   (3,3) 基础矩阵
*/
Eigen::Matrix3f Stereo::fundamentalEightPoint(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2)
{
    assert(p1.size() == p2.size());
    int pt_N = p1.cols();
    
    Eigen::Matrix3f F;
    
    Eigen::Matrix<float, 9, 1> F_vec;
    Eigen::Matrix<float, Eigen::Dynamic, 9> A;
    A.resize(pt_N, 9);

    // 克罗内克积，构造A
    for(int i =0; i < pt_N; i++)
    {
        A.row(i) = mango::kronecker<float>(p1.col(i), p2.col(i)).transpose();
    }

    // SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf U, S, V;
    U = svd.matrixU();
    V = svd.matrixV();
    S = svd.singularValues();
    F_vec = V.col(V.cols() - 1);

    // 恢复矩阵形式，行列式为0约束
    Eigen::Matrix3f F_mat;
    F_mat.row(0) = F_vec.block(0,0,3,1).transpose();
    F_mat.row(1) = F_vec.block(3,0,3,1).transpose();
    F_mat.row(2) = F_vec.block(6,0,3,1).transpose();
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_F(F_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd_F.matrixU();
    V = svd_F.matrixV();
    S = svd_F.singularValues();
    // 最小奇异值设为0，满足行列式值为0的约束，同时尽量使误差小
    Eigen::Matrix3f S_mat;
    S_mat << S(0), 0, 0,
             0, S(1), 0,
             0,    0, 0;
    F = U * S_mat * V.transpose();

    return F;
}


/**
 * 归一化八点法计算基础矩阵
 * @param p1 (3,N) 图像一中的像素点齐次坐标
 * @param p2 (3,N) 图像二中的像素点齐次坐标
 * @return   (3,3) 基础矩阵
*/
Eigen::Matrix3f Stereo::fundamentalEightPointNormalized(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2)
{
    Eigen::Matrix3f F;

    Eigen::Matrix<float, 3, Eigen::Dynamic> p1_normalized, p2_normalized;
    Eigen::Matrix3f T1, T2;
    normalizePoint(p1, p1_normalized, T1);
    normalizePoint(p2, p2_normalized, T2);

    F = fundamentalEightPoint(p1_normalized, p2_normalized);
    F = T2.transpose() * F * T1;

    return F;
}


/**
 * 像素点归一化
 * @param pt           (3,N) 像素坐标
 * @param pt_normalized (3,N) 归一化像素坐标
 * @param transform    (3,3) 变换矩阵
*/
void Stereo::normalizePoint(const Eigen::Matrix<float, 3, Eigen::Dynamic>& pt, Eigen::Matrix<float, 3, Eigen::Dynamic>& pt_normalized, Eigen::Matrix3f& transform)
{
    int n = pt.cols();
    Eigen::VectorXf mean = pt.rowwise().mean();
    // std::cout << "mean:" << mean << std::endl;
    // std::cout << "pt:" << pt.block(0,0,3,10) << std::endl;
    float sum_d = 0;
    for(int i = 0; i < n; i++)
    {
        float d = pow(pt(0,i) - mean(0), 2) + pow(pt(1,i) - mean(1), 2);
        sum_d += d;
    }
    float sigma = sqrt(sum_d / n);
    // std::cout << "sigma:" << sigma << std::endl;
    float s = sqrt(2) / sigma;

    transform << s, 0, -s*mean(0),
                 0, s, -s*mean(1),
                 0, 0,          1;
    
    pt_normalized = transform * pt;
    // std::cout << "pt_norm:" << pt_normalized.block(0,0,3,10) << std::endl;
}

/**
 * 计算本质矩阵
 * @param p1 (3,N) 图像一中的像素点齐次坐标
 * @param p2 (3,N) 图像二中的像素点齐次坐标
 * @param K1 (3,3) 相机内参一
 * @param K2 (3,3) 相机内参二
 * @return   (3,3) 本质矩阵
*/
Eigen::Matrix3f Stereo::estimateEssentialMatrix(const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1,
                                        const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2,
                                        const Eigen::Matrix3f& K1,
                                        const Eigen::Matrix3f& K2)
{
    Eigen::Matrix3f E;

    Eigen::MatrixXf F = fundamentalEightPointNormalized(p1, p2);
    E = K2.transpose() * F * K1;

    return E;
}

/**
 * 点到极线距离之和
 * @param F  (3,3) 基础矩阵
 * @param p1 (3,N) 图像一中的像素点齐次坐标
 * @param p2 (3,N) 图像二中的像素点齐次坐标
 * @return         距离之和
*/
double Stereo::distPoint2EpipolarLine(const Eigen::Matrix3f& F, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1, const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2)
{
    assert(p1.cols() == p2.cols());

    int pt_N = p1.cols();
    Eigen::MatrixXf line1 = F.transpose() * p2;
    Eigen::MatrixXf line2 = F * p1;
    
    double sum;
    for(int i = 0; i < pt_N; i++)
    {
        double a = pow(line1.col(i).transpose() * p1.col(i), 2);
        double b = pow(line1(0,i), 2) + pow(line1(1,i), 2);
        sum += a/b;

        a = pow(line2.col(i).transpose() * p2.col(i), 2);
        b = pow(line2(0,i), 2) + pow(line2(1,i), 2);
        sum += a/b;
    }

    return sqrt(sum / pt_N);
}

/**
 * 本质矩阵分解
 * @param rotation 两个(3,3)的旋转矩阵
 * @param u3       平移信息
*/
void Stereo::decomposeEssentialMatrix(const Eigen::Matrix3f& E, Eigen::Matrix<float, 6, 3>& rotation, Eigen::Vector3f& u3)
{
    // SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXf U, S, V;
    U = svd.matrixU();
    V = svd.matrixV();
    S = svd.singularValues();
    
    // U的第三列
    u3 = U.col(2);
    if(u3.norm() != 0)
    {
        u3.normalize();
    }

    // 两个旋转矩阵
    Eigen::Matrix3f W;
    W << 0, -1, 0,
         1, 0, 0,
         0, 0, 1;
    Eigen::Matrix3f R1 = U * W * V.transpose();
    if(R1.determinant() < 0)
    {
        R1 *= -1;
    }

    Eigen::Matrix3f R2 = U * W.transpose() * V.transpose();
    if(R2.determinant() < 0)
    {
        R2 *= -1;
    }

    rotation.block(0,0,3,3) = R1;
    rotation.block(3,0,3,3) = R2;
}

/**
 * 通过三角化选择正确的位姿结果
 * @param rotation  (6,3) 两个(3,3)的旋转矩阵
 * @param u3        (3,1) 平移
 * @param p1        (3,N) 图像一中的像素点齐次坐标
 * @param p2        (3,N) 图像二中的像素点齐次坐标
 * @param K1        (3,3) 相机内参一
 * @param K2        (3,3) 相机内参二
 * @return          (3,4) [R|t]
*/
Eigen::Matrix<float, 3, 4> Stereo::disambiguateRelativePose(const Eigen::Matrix<float, 6, 3>& rotation, 
                                                            const Eigen::Vector3f& u3,
                                                            const Eigen::Matrix<float, 3, Eigen::Dynamic>& p1,
                                                            const Eigen::Matrix<float, 3, Eigen::Dynamic>& p2,
                                                            const Eigen::Matrix3f& K1,
                                                            const Eigen::Matrix3f& K2)
{
    Eigen::Matrix<float, 3, 4> T;

    // 视角一的投影矩阵
    Eigen::Matrix<float, 3, 4> M1 = K1 * Eigen::Matrix<float, 3, 4>::Identity();
    Eigen::Matrix<float, 3, 4> M2, T_21;

    int max_pt_N = 0;

    // 一共四种组合
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 2; j++)
        {
            // [R|t]
            Eigen::Matrix3f R = rotation.block(i*3,0,3,3);
            Eigen::Vector3f t = u3 * pow(-1, j);
            T_21.block(0,0,3,3) = R;
            T_21.block(0,3,3,1) = t;

            // 计算投影矩阵
            M2 = K2 * T_21;

            // 三角化，P_1为点在图一的相机坐标系下的表示，P_2为点在图二的相机坐标系下的表示
            Eigen::Matrix4Xf P_1 = linearTriangulation(p1, p2, M1, M2);
            Eigen::Matrix3Xf P_2 = T_21 * P_1;

            // 统计z值为正的点数
            int N = 0;
            for(int i = 0; i < P_1.cols(); i++)
            {
                if(P_1(2,i) > 0)
                {
                    N++;
                }
                if(P_2(2,i) > 0)
                {
                    N++;
                }
            }
            // 选择三角化点数最多的位姿矩阵
            if(N > max_pt_N)
            {
                max_pt_N = N;
                T = T_21;
            }
        }
    }
}
}