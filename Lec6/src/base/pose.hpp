#pragma once
#include <eigen3/Eigen/Core>

namespace mango {

template<class Ttype>
class _Pose3D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    _Pose3D() = default;
    
    _Pose3D(const Ttype& wx, 
           const Ttype& wy,
           const Ttype& wz, 
           const Ttype& tx,
           const Ttype& ty,
           const Ttype& tz)
        : wx(wx), wy(wy), wz(wz), tx(tx), ty(ty), tz(tz)
    {
        R_ = AxisAngle2RotationMatrix(wx, wy, wz);
        t_ = Eigen::Matrix<Ttype, 3, 1>(tx, ty, tz);
    }

    _Pose3D(const Eigen::Matrix<Ttype, 3, 3>& R, const Eigen::Matrix<Ttype, 3, 1>& t)
        : R_(R), t_(t)
    {
        tx = t(0);
        ty = t(1);
        tz = t(2);
    }

    ~_Pose3D() {}

    _Pose3D& operator=(const _Pose3D& rhs)
    {
        wx = rhs.wx;
        wy = rhs.wy;
        wz = rhs.wz;
        tx = rhs.tx;
        ty = rhs.ty;
        tz = rhs.tz;
        R_ = rhs.R_;
        t_ = rhs.t_;
        return *this;
    }

    _Pose3D(const _Pose3D& rhs)
    {
        wx = rhs.wx;
        wy = rhs.wy;
        wz = rhs.wz;
        tx = rhs.tx;
        ty = rhs.ty;
        tz = rhs.tz;
        R_ = rhs.R_;
        t_ = rhs.t_;
    }

    /**
     * 轴角转换成旋转矩阵
    */
    inline Eigen::Matrix<Ttype, 3, 3> AxisAngle2RotationMatrix(Ttype wx, Ttype wy, Ttype wz)
    {
        Ttype mod = sqrtl(wx * wx + wy * wy + wz * wz);
        Eigen::Matrix<Ttype, 3, 3> _R, k;
        k <<          0, -wz/mod, wy/mod,
                wz/mod,        0, -wx/mod,
             -wy/mod, wx/mod,         0;
        _R = Eigen::Matrix<Ttype, 3, 3>::Identity() + sin(mod) * k + (1 - cos(mod)) * k * k;

        return _R;
    }

    /**
     * 旋转矩阵R
    */
    inline Eigen::Matrix<Ttype, 3, 3> R() const
    {
        return R_;
    }

    /**
     * 平移向量t
    */
    inline Eigen::Matrix<Ttype, 3, 1> t() const
    {
        return t_;
    }

    /**
     * 变换矩阵[R t]
    */
    inline Eigen::Matrix<Ttype, 4, 4> Rt() const
    {
        Eigen::Matrix<Ttype, 4, 4> _T = Eigen::Matrix<Ttype, 4, 4>::Zero();
        _T.block(0, 0, 3, 3) = R();
        _T.block(0, 3, 3, 1) = t();
        _T(3, 3) = 1;
        
        return _T;
    }

    /**
     * 坐标轴原点
    */
    inline Eigen::Matrix<Ttype, 3, 1>  origin() const
    {
        Eigen::Matrix<Ttype, 3, 1> p = -R_.inverse() * t_;
        return p;
    }

    // (wx, wy, wz) 轴角表示，向量表示旋转轴，模长表示旋转角
    Ttype wx, wy, wz, tx, ty, tz;
    Eigen::Matrix<Ttype, 3, 3> R_;
    Eigen::Matrix<Ttype, 3, 1> t_;
};

using Pose3D = _Pose3D<double>;
}