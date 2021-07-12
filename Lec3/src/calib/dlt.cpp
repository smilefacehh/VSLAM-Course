#include "dlt.h"

#include <iostream>
#include <assert.h>

namespace mango {

DLT::DLT(const Eigen::Matrix<double, Eigen::Dynamic, 3>& world_pts, const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs)
{
    world_pts_ = world_pts;
    pxs_ = pxs;
    camera_K_ = Eigen::Matrix3d::Zero();
    assert(world_pts_.rows() == pxs_.rows());
    pt_num_ = world_pts_.rows();
}
DLT::DLT(const Eigen::Matrix<double, Eigen::Dynamic, 3>& world_pts, const Eigen::Matrix<double, Eigen::Dynamic, 2>& pxs, const Eigen::Matrix3d& camera_K)
{
    world_pts_ = world_pts;
    pxs_ = pxs;
    camera_K_ = camera_K;
    assert(world_pts_.rows() == pxs_.rows());
    pt_num_ = world_pts_.rows();
}

DLT::~DLT() {}

void DLT::run()
{
    Eigen::Matrix<double, Eigen::Dynamic, 12> Q = makeQ();
    Eigen::Matrix<double, 3, 4> M = getM(Q);
    if(camera_K_ == Eigen::Matrix3d::Zero())
    {
        decompM(M, pose_, camera_K_);
    }
    else
    {
        pose_ = poseFromM(M);
    }
}

Eigen::Matrix<double, Eigen::Dynamic, 12> DLT::makeQ()
{
    Eigen::Matrix<double, Eigen::Dynamic, 12> Q;
    Q.resize(pt_num_ * 2, 12);

    if(camera_K_ == Eigen::Matrix3d::Zero())
    {
        for(int i = 0; i < pt_num_; i++)
        {
            Eigen::Matrix<double, 12, 1> row;
            row << world_pts_(i, 0), world_pts_(i, 1), world_pts_(i, 2), 1, 0, 0, 0, 0, -pxs_(i, 0) * world_pts_(i, 0), -pxs_(i, 0) * world_pts_(i, 1), -pxs_(i, 0) * world_pts_(i, 2), -pxs_(i, 0);
            Q.row(2 * i) = row;
            row << 0, 0, 0, 0, world_pts_(i, 0), world_pts_(i, 1), world_pts_(i, 2), 1, -pxs_(i, 1) * world_pts_(i, 0), -pxs_(i, 1) * world_pts_(i, 1), -pxs_(i, 1) * world_pts_(i, 2), -pxs_(i, 1);
            Q.row(2 * i + 1) = row;
        }
    }
    else
    {
        for(int i = 0; i < pt_num_; i++)
        {
            Eigen::Vector3d inverse_K_px = camera_K_.inverse() * Eigen::Vector3d(pxs_(i, 0), pxs_(i, 1), 1);
            Eigen::Matrix<double, 12, 1> row;
            row << world_pts_(i, 0), world_pts_(i, 1), world_pts_(i, 2), 1, 0, 0, 0, 0, -inverse_K_px(0) * world_pts_(i, 0), -inverse_K_px(0) * world_pts_(i, 1), -inverse_K_px(0) * world_pts_(i, 2), -inverse_K_px(0);
            Q.row(2 * i) = row;
            row << 0, 0, 0, 0, world_pts_(i, 0), world_pts_(i, 1), world_pts_(i, 2), 1, -inverse_K_px(1) * world_pts_(i, 0), -inverse_K_px(1) * world_pts_(i, 1), -inverse_K_px(1) * world_pts_(i, 2), -inverse_K_px(1);
            Q.row(2 * i + 1) = row;
        }
    }
    return Q;
}

Eigen::Matrix<double, 3, 4> DLT::getM(const Eigen::Matrix<double, Eigen::Dynamic, 12> Q)
{
    // Q = USV^T，这里的V就是公式里的V
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U, S, V;
    U = svd.matrixU();
    V = svd.matrixV();
    S = svd.singularValues();
    Eigen::VectorXd solu = V.col(V.cols() - 1);
    Eigen::Matrix<double, 3, 4> M;
    M << solu(0), solu(1),  solu(2),  solu(3),
         solu(4), solu(5),  solu(6),  solu(7),
         solu(8), solu(9), solu(10), solu(11);
    
    // 确保R的行列式是+1，旋转矩阵的要求
    if(M.block(0, 0, 3, 3).determinant() < 0)
    {
        M *= -1;
    }

    return M;
}

Pose3D DLT::poseFromM(const Eigen::Matrix<double, 3, 4> M)
{
    Eigen::Matrix3d R = M.block(0, 0, 3, 3);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U, S, V;
    U = svd.matrixU();
    V = svd.matrixV();
    S = svd.singularValues();

    double d0 = R.norm();
    R = U * V.transpose();

    // 用norm，不用squaredNorm
    double alpha = R.norm() / d0;
    Eigen::Vector3d t = M.block(0, 3, 3, 1);
    t *= alpha;

    Pose3D pose(R, t);

    return pose;
}

void DLT::decompM(const Eigen::Matrix<double, 3, 4> M, Pose3D& pose, Eigen::Matrix3d& K)
{
    Eigen::Matrix3d KR = M.block(0, 0, 3, 3);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr;
    qr.compute(KR);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    Eigen::MatrixXd Q = qr.householderQ();

    K = R;
    K /= K(2, 2);
    
    Eigen::Vector3d Kt = M.block(0, 3, 3, 1);
    Eigen::Vector3d t = K.inverse() * Kt;

    pose.R_ = Q;
    pose.t_ = t;
}
}