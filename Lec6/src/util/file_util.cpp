#include "file_util.h"

#include <iostream>

namespace mango {

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> load_matrix(const std::string& file, int row, int col)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
    mat.resize(row, col);

    std::ifstream fs(file.c_str());
    if(!fs.is_open())
    {
        std::cerr << "failed to load file: " << file << std::endl;
        return mat;
    }
    for(int r = 0; r < row; r++)
    {
        for(int c = 0; c < col; c++)
        {
            fs >> mat(r, c);
        }
    }

    fs.close();
    return mat;
}

}