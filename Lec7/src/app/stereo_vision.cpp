#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>


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
    try
    {
        std::string imgfile = data_folder_+ "images.txt";
        
    }
    catch(std::exception e)
    {
        std::cerr << "image process exception: " << e.what() << std::endl;
    }
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