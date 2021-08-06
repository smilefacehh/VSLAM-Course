#pragma once

#include <eigen3/Eigen/Core>

namespace mango {

template<class T>
class _Point2D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    _Point2D() = default;
    _Point2D(const T& x, const T& y): x(x), y(y) {}
    _Point2D(const Eigen::Matrix<T, 2, 1>& p): x(p(0, 0)), y(p(1, 0)) {}
    ~_Point2D() {}

    Eigen::Matrix<T, 2, 1> eigenType()
    {
        Eigen::Matrix<T, 2, 1> p(x, y);
        return p;
    }

    T x, y;
};

template<class T>
class _Point3D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    _Point3D() = default;
    _Point3D(const T& x, const T& y, const T& z): x(x), y(y), z(z) {}
    _Point3D(const Eigen::Matrix<T, 3, 1>& p): x(p(0, 0)), y(p(1, 0)), z(p(2, 0)) {}
    ~_Point3D() {}

    Eigen::Matrix<T, 3, 1> eigenType()
    {
        Eigen::Matrix<T, 3, 1> p(x, y, z);
        return p;
    }

    T x, y, z;
};

using Point2D = _Point2D<double>;
using Point3D = _Point3D<double>;
using Point2Df = _Point2D<float>;
using Point3Df = _Point3D<float>;
}