#ifndef MY_STEREO_BM_H
#define MY_STEREO_BM_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class MyStereoBM {
    public:
        class State {
            public:
                int max_disparity;
                int window_size;
                int window_size2;
                int width;
                int height;
                double focal_length;
                double baseline;
                double cx1;
                double cx2;
                double cy1;
                double cy2;
                double doffs;
                
                void scale(int factor) {
                    max_disparity /= factor;
                    window_size /= factor;
                    window_size2 /= factor;
                    width /= factor;
                    height /= factor;
                }
        };

        MyStereoBM(State state);
        cv::Mat compute(cv::Mat left, cv::Mat right);
        cv::Mat computeDynamic(cv::Mat left, cv::Mat right);

    private:
        cv::Mat drawDisparity(
                const std::vector<std::vector<int>>& disparity_map);
        int SumOfAbsoluteDifferences(
                cv::Mat left,
                cv::Point p1,
                cv::Mat right,
                cv::Point p2);
        double disparityToMM(int disparity);

        // deprecated, used with normalized correlation
        void getDisparityWindow(
                cv::Mat img,
                cv::Point origin,
                std::vector<uchar>& out);

        State s;
        const cv::Point offset = cv::Point(0, 0);
        const static int SCALE = 2;
};

#endif
