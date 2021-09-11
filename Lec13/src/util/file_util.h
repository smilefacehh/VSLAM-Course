#pragma once

#include <fstream>
#include <string>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>

namespace mango {

/**
 * 从文本文件加载矩阵，仅限于空格隔开
 * @param file 文件路径
 * @param row  行数
 * @param col  列数
*/
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> load_matrix(const std::string& file, int row, int col);

}