#include "MyStereoBM.h"

#include <iostream>
#include <algorithm>

using namespace std;
using namespace cv;

MyStereoBM::MyStereoBM(MyStereoBM::State state) {
    state.window_size2 = state.window_size/2;
    state.doffs = state.cx2 - state.cx1;
    this->s = state;
}

Mat MyStereoBM::computeDynamic(Mat left_src, Mat right_src) {
    Mat left;
    Mat right;
    resize(left_src, left, Size(), 1.0/SCALE, 1.0/SCALE);
    resize(right_src, right, Size(), 1.0/SCALE, 1.0/SCALE);
    s.scale(SCALE);

    vector<vector<int>> disparity_map(s.width, vector<int>(s.height));
    vector<vector<int>> disparity_cost(s.width, vector<int>(s.max_disparity));
    vector<vector<int>> optimal_indices(s.width, vector<int>(s.max_disparity));
    for (int y = offset.y; y < s.height; y++) {
        // progress output
        cout << y << " / " << s.height << endl;

        for(auto& i: disparity_cost) {
            std::fill(i.begin(), i.end(), INT_MAX);
        }
        for(auto& i: optimal_indices) {
            std::fill(i.begin(), i.end(), 0);
        }

        // row bounds for block
        int minr = max(0, y - s.window_size2);
        int maxr = min(s.height, y + s.window_size2);

        for (int x = offset.x; x < s.width; x++) {

            // column bounds for the block
            int minc = max(0, x - s.window_size2);
            int maxc = min(s.width, x + s.window_size2);

            // how many pixels we can look left and right
            int mind = max(-s.max_disparity, -minc);
            //int maxd = min(s.max_disparity, s.width - maxc);
            int maxd = 0;

            for (int i = mind; i < maxd; i++) {
                disparity_cost[y][i + s.max_disparity] = 
                    SumOfAbsoluteDifferences(left, Point(x, y),
                            right, Point(i + x));
            }

            vector<int>& last_row = optimal_indices[y];
            for (int i = 0; i < s.width; i++) {
                int finf = 1000; // false infinity
                int cfinf = i * finf;
                int disparity_penalty = 0.5;


                vector<vector<int>> penalized;
                vector<int> v;
                // i cant figure this out :(
                // you need to like fit every pixel and determine
                // when it is best to incur a disparity penalty
                // when there is no match nearby
            }
        }
    }

    return drawDisparity(disparity_map);
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
    writeFilePFM(disparity_map, "disp0MUNROE.pfm", 1.0/s.max_disparity);
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


int MyStereoBM::littleEndian() {
    int intval = 1;
    uchar *uval = (uchar *)&intval;
    return uval[0] == 1;
}

// adapted from Middlebury evaluation code
void MyStereoBM::writeFilePFM(const vector<vector<int>>& data,
        const char* filename, float scale_factor=1/255.0) {
    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        fprintf(stderr, "writeFilePFM: could not open %s\n", filename);
        exit(1);
    }

    if (littleEndian()) {
        scale_factor = -scale_factor;
    }

    fprintf(stream, "Pf\n%d %d\n%f\n", s.width, s.height, scale_factor);

    int n = s.width;
    // write rows -- pfm stores rows in inverse order!
    for (int y = s.height-1; y >= 0; y--) {
        for (int x = 0; x < s.width; x++) {
            float val = data[x][y];
            if ((int)fwrite(&val, sizeof(float), 1, stream) != 1) {
                fprintf(stderr, "WriteFilePFM: problem writing data\n");
                exit(1);
            }
        }
        
    }

    // close file
    fclose(stream);
}
