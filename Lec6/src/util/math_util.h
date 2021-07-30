#pragma once

#include <vector>

namespace mango {

/**
 * 计算SSD误差
 * 注：模板函数的实现要放到头文件中
*/
template <typename T>
double ssd(const std::vector<T>& v1, const std::vector<T>& v2)
{
    assert(v1.size() == v2.size());

    double sum = 0;

    for(int i = 0; i < v1.size(); i++)
    {
        double d = v1[i] - v2[i];
        sum += d * d;
    }

    return sum;
}

}