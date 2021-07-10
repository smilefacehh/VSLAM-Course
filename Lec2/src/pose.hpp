#pragma once
#include <eigen3/Eigen/Core>

namespace mango {

template<class Ttype>
class _Pose3D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    _Pose3D(const Ttype& wx, 
           const Ttype& wy,
           const Ttype& wz, 
           const Ttype& tx,
           const Ttype& ty,
           const Ttype& tz)
        : wx(wx), wy(wy), wz(wz), tx(tx), ty(ty), tz(tz)
    {}

    ~_Pose3D() {}

    _Pose3D& operator=(const _Pose3D& rhs)
    {
        wx = rhs.wx;
        wy = rhs.wy;
        wz = rhs.wz;
        tx = rhs.tx;
        ty = rhs.ty;
        tz = rhs.tz;
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
    }

    /**
     * 旋转矩阵R
    */
    inline Eigen::Matrix<Ttype, 3, 3> R() const
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
     * 平移向量t
    */
    inline Eigen::Matrix<Ttype, 3, 1> t() const
    {
        return Eigen::Matrix<Ttype, 3, 1>(tx, ty, tz);
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

    // (wx, wy, wz) 轴角表示，向量表示旋转轴，模长表示旋转角
    Ttype wx, wy, wz, tx, ty, tz;
};

using Pose3D = _Pose3D<double>;
}