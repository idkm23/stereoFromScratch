#include "MyStereoBM.h"

#include <iostream>

using namespace std;
using namespace cv;

MyStereoBM::MyStereoBM(MyStereoBM::State state) {
    state.window_size2 = state.window_size/2;
    state.doffs = state.cx2 - state.cx1;
    this->s = state;
}

Mat MyStereoBM::compute(Mat left_src, Mat right_src) {
    Mat left;
    Mat right;
    resize(left_src, left, Size(), 1.0/SCALE, 1.0/SCALE);
    resize(right_src, right, Size(), 1.0/SCALE, 1.0/SCALE);
    s.scale(SCALE);

    vector<vector<int>> disparity_map(s.width, vector<int>(s.height));
    for (int y = offset.y; y < s.height; y++) {
        // progress output
        cout << y << " / " << s.height << endl;
        for (int x = offset.x; x < s.width; x++) {
            int min_sad = -1;
            int best_x = -1;
            int scan_x = x - s.max_disparity;
            if (scan_x < offset.x) {
                scan_x = offset.x;
            }
            int end_scan_x = x;
            // scan to the right
            //int end_scan_x = x + s.max_disparity;
            //if (end_scan_x > s.width) {
            //    end_scan_x = s.width;
            //}
            while(scan_x < end_scan_x) {
                int sad =
                    SumOfAbsoluteDifferences(left, Point(x, y),
                            right, Point(scan_x, y));
                // penalize the SAD if disparity is
                // too different from disparity_map[x][y-3:y-1]
                if (min_sad == -1 || sad < min_sad) {
                    min_sad = sad;
                    best_x = scan_x;
                }
                scan_x++;
            }
            if (best_x != -1) {
                disparity_map[x][y] = abs(best_x - x);
            } else {
                disparity_map[x][y] = -1;
            }
        }
    }

    return drawDisparity(disparity_map);
}

Mat MyStereoBM::drawDisparity(const vector<vector<int>>& disparity_map) {
    Mat small_out(s.height, s.width, CV_8UC1);
    Mat out;
    double mag = 0;
    for (int y = offset.y; y < s.height; y++) {
        for (int x = offset.x; x < s.width; x++) {
            cout << x+y << " " << disparity_map[x][y] << endl;
            mag = disparity_map[x][y]/(double)s.max_disparity;
            if (mag > 1) {
                mag = 1;
            }
            small_out.at<uchar>(y, x) = 255 * mag;
        }
    }
    resize(small_out, out, Size(), SCALE, SCALE);
    applyColorMap(out, out, cv::COLORMAP_JET);
    return out;
}


// block match rating, fast
int MyStereoBM::SumOfAbsoluteDifferences(Mat left, Point p1, Mat right, Point p2) {
    int sum = 0;
    Point orig(p1.x - s.window_size2, p1.y - s.window_size2);
    Point scan(p2.x - s.window_size2, p2.y - s.window_size2);
    for (int x = 0; x < s.window_size; x++) {
        for (int y = 0; y < s.window_size; y++) {
            if (orig.x >= 0 && orig.x < s.width
                    && scan.x >= 0 && scan.x < s.width
                    && orig.y >= 0 && orig.y < s.height
                    && scan.y >= 0 && scan.y < s.height) {
                sum += abs(left.at<uchar>(orig.y, orig.x)
                        - right.at<uchar>(scan.y, scan.x));
            }
            orig.y++;
            scan.y++;
        }
        orig.x++;
        scan.x++;
        orig.y -= s.window_size;
        scan.y -= s.window_size;
    }
    return sum;
}

// deprecated, used with normalized correlation
void MyStereoBM::getDisparityWindow(Mat img, Point origin,
        vector<uchar>& out) {
    int real_x, real_y;
    for (int x = 0; x < s.window_size; x++) {
        real_x = origin.x + x - s.window_size2;
        for (int y = 0; y < s.window_size; y++) {
            real_y = origin.y + y - s.window_size/2;
            if (real_x >= 0 && real_x < s.width
                    && real_y >= 0 && real_y < s.height) {
                out.push_back(img.at<uchar>(real_x, real_y));
            }
        }
    }
}

double MyStereoBM::disparityToMM(int disparity) {
    return s.baseline * s.focal_length / (disparity*SCALE + s.doffs);
}
