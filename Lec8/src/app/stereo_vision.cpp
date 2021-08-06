#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "util/string_util.h"

namespace mango {

class StereoVision
{
public:
    StereoVision(const std::string& data_folder)
        : data_folder_(data_folder)
    {}
    ~StereoVision() {}

    void run();

private:
    std::string data_folder_;
};

void StereoVision::run()
{
    
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