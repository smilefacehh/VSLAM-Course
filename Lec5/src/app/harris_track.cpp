#include <iostream>

#include <opencv2/opencv.hpp>

#include "../util/string_util.h"
#include "../feature/harris.h"
#include "../util/util.h"
#include "../util/draw_util.h"

namespace mango {

class HarrisTrack
{
public:
    HarrisTrack(const std::string& data_folder)
        : data_folder_(data_folder)
    {}
    ~HarrisTrack() {}

    void run();

private:
    std::string data_folder_;
};

void HarrisTrack::run()
{
    try
    {
        std::string img_file_path = data_folder_ + "images.txt";
        std::ifstream fs(img_file_path.c_str());
        if(!fs.is_open())
        {
            std::cerr << "can't open file:" << img_file_path << std::endl;
            return;
        }

        std::vector<cv::Point2i> last_frame_kps;
        std::vector<std::vector<uchar>> last_frame_descs;
        while(!fs.eof())
        {
            std::string img_path;
            fs >> img_path;
            img_path = data_folder_ + img_path;
            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

            mango::Harris harris;
            cv::Mat resp;
            std::vector<cv::Point2i> kps;
            std::vector<std::vector<uchar>> descs;
            std::vector<int> match_idx;
            harris.detect(img, resp, 3, 9, 0.08);
            harris.getKeyPoints(resp, kps, 200, 8);
            harris.getDescriptors(img, kps, descs, 9);

            cv::Mat match_img;
            if(last_frame_descs.size() != 0)
            {
                harris.match(last_frame_descs, descs, match_idx, 5);
                match_img = harris.plotMatchOneImage(img, last_frame_kps, kps, match_idx);
            }

            last_frame_descs = descs;
            last_frame_kps = kps;

            cv::Mat kp_img = mango::drawPoint<cv::Point2i>(resp, kps, mango::DrawType::CIRCLE, cv::Scalar(0,0,255));

            std::vector<cv::Mat> merged_imgs;
            if(match_img.empty())
            {
                merged_imgs.push_back(img);
                merged_imgs.push_back(kp_img);
            }
            else
            {
                merged_imgs.push_back(img);
                merged_imgs.push_back(match_img);
                merged_imgs.push_back(kp_img);
            }
            cv::Mat merge = mango::mergeImage(merged_imgs, img.cols/2, img.rows);
            cv::imshow("merge", merge);
            cv::waitKey(30);
        }
        fs.close();
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
        std::cout << "Usage: harris_track ../data" << std::endl;
        return 0;
    }

    std::string data_folder = mango::folder_add_slash(argv[1]);

    mango::HarrisTrack harris_track(data_folder);
    harris_track.run();

    return 0;
}