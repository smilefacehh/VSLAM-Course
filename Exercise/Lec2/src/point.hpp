#pragma once

namespace mango {

template<class T>
class _Point2D
{
public:
    _Point2D() = default;
    _Point2D(const T& x, const T& y): x(x), y(y) {}
    ~_Point2D() {}

    T x, y;
};

template<class T>
class _Point3D
{
public:
    _Point3D() = default;
    _Point3D(const T& x, const T& y, const T& z): x(x), y(y), z(z) {}
    ~_Point3D() {}

    T x, y, z;
};

using Point2D = _Point2D<double>;
using Point3D = _Point3D<double>;
using Point2Df = _Point2D<float>;
using Point3Df = _Point3D<float>;
}