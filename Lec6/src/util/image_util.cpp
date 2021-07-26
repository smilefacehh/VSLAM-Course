#include "image_util.h"

namespace mango
{

/**
 * 合并图像
 * @param imgs 图像
 * @param w    图像缩放为指定大小，宽
 * @param h    图像缩放为指定大小，高
*/
cv::Mat mergeImage(const std::vector<cv::Mat>& imgs, int w, int h)
{
    int n = (int)imgs.size();
    int merge_w, merge_h;
    if(n <= 3)
    {
        merge_w = w * n;
        merge_h = h;
    }
    else if(n % 2 == 0)
    {
        merge_w = w * n / 2;
        merge_h = h * 2;
    }
    else
    {
        merge_w = w * n / 2 + 1;
        merge_h = h * 2;
    }

    cv::Mat merge_img(merge_h, merge_w, CV_8UC3);
    for(int i = 0; i < n; i++)
    {
        int row, col;
        if(n <= 3)
        {
            row = 0;
            col = i;
        }
        else if(n % 2 == 0)
        {
            row = i < n / 2 ? 0 : 1;
            col = i % (n / 2);
        }
        else
        {
            row = i < (n / 2 + 1) ? 0 : 1;
            col = i % (n / 2 + 1);
        }

        cv::Mat tmp;
        cv::resize(imgs[i], tmp, cv::Size(w, h));

        cv::Mat img_colorful(h, w, CV_8UC3);
        if(tmp.channels() == 1)
        {
            std::vector<cv::Mat> channels(3, tmp);
            cv::merge(channels, img_colorful);
        }
        else
        {
            tmp.copyTo(img_colorful);
        }

        img_colorful.copyTo(merge_img(cv::Rect(col * w, row * h, w, h)));
    }
    return merge_img;
}

}