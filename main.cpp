#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <string>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

// gray has one channel, with a range of 0-255

void PrintVector(const vector<uchar>& vec) {
    cout << "{ ";
    for(auto i = vec.begin(); i != vec.end(); ++i) {
        cout << (int)*i << " ";
    }
    cout << "}\n";
}

// https://arxiv.org/pdf/1409.3024.pdf
// block match rating, slow but robust to outliers
double NormalizedCorrelation(const vector<uchar>& img1, vector<uchar> img2) {
    long sum = 0;
    long energy_1 = 0;
    long energy_2 = 0;
    for (int i = 0; i < img1.size(); i++) {
        sum += img1[i] * img2[i];
        energy_1 += img1[i] * img1[i];
        energy_2 += img2[i] * img2[i];
    }
    return sum / sqrt(energy_1 * energy_2);
}

class MyStereo {
    public:
        void drawDisparity(Mat img1, Mat img2, Mat out) {
            vector<vector<int>> disparity_map(width, vector<int>(height));
            for (int y = offset.y; y < height; y++) {
                cout << y << " / " << height << endl;
                for (int x = offset.x; x < width; x++) {
                    int min_sad = -1;
                    int best_x = -1;
                    //int scan_x = x;
                    int scan_x = x - max_disparity;
                    if (scan_x < offset.x) {
                        scan_x = offset.x;
                    }
                    int end_scan_x = x;
                    //int end_scan_x = x + max_disparity;
                    //if (end_scan_x > width) {
                    //    end_scan_x = width;
                    //}
                    while(scan_x < end_scan_x) {
                        int sad =
                            SumOfAbsoluteDifferences(img1, Point(x, y),
                                    img2, Point(scan_x, y));
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
                    //cout << disparity_map[x][y] << endl;
                }
            }
            double mag = 0;
            for (int y = offset.y; y < height; y++) {
                for (int x = offset.x; x < width; x++) {
                    cout << x+y << " " << disparity_map[x][y] << endl;
                    mag = disparity_map[x][y]/(double)max_disparity;
                    if (mag > 1) {
                        mag = 1;
                    }
                    out.at<uchar>(y, x) = 255 * mag;
                }
            }
        }


        // block match rating, fast
        int SumOfAbsoluteDifferences(Mat img1, Point p1, Mat img2, Point p2) {
            int sum = 0;
            Point orig(p1.x - HWINDOW_SIZE, p1.y - HWINDOW_SIZE);
            Point scan(p2.x - HWINDOW_SIZE, p2.y - HWINDOW_SIZE);
            for (int x = 0; x < WINDOW_SIZE; x++) {
                for (int y = 0; y < WINDOW_SIZE; y++) {
                    if (orig.x >= 0 && orig.x < width
                            && scan.x >= 0 && scan.x < width
                            && orig.y >= 0 && orig.y < height
                            && scan.y >= 0 && scan.y < height) {
                        sum += abs(img1.at<uchar>(orig.y, orig.x)
                                - img2.at<uchar>(scan.y, scan.x));
                        //cout << "\no x " << orig.x << " y " << orig.y;
                        //cout << "\ns x " << scan.x << " y " << scan.y;
                    }
                    //else {
                    //    cout << "\nEo x " << orig.x << " y " << orig.y;
                    //    cout << "\nEs x " << scan.x << " y " << scan.y;
                    //}
                    orig.y++;
                    scan.y++;
                }
                orig.x++;
                scan.x++;
                orig.y -= WINDOW_SIZE;
                scan.y -= WINDOW_SIZE;
            }
            return sum;
        }

        // deprecated, used with normalized correlation
        void getDisparityWindow(Mat img, Point origin,
                vector<uchar>& out) {
            int real_x, real_y;
            for (int x = 0; x < WINDOW_SIZE; x++) {
                real_x = origin.x + x - WINDOW_SIZE/2;
                for (int y = 0; y < WINDOW_SIZE; y++) {
                    real_y = origin.y + y - WINDOW_SIZE/2;
                    if (real_x >= 0 && real_x < width
                            && real_y >= 0 && real_y < height) {
                        out.push_back(img.at<uchar>(real_x, real_y));
                    }
                }
            }
        }

        double disparityToMM(int disparity) {
            return baseline * focal_length / (disparity + doffs);
        }

        // one side of the scanning window
        const static int WINDOW_SIZE = 7;
        const static int HWINDOW_SIZE = WINDOW_SIZE/2;

        const Point offset = Point(0, 0);
        const int width = 450;//2880;
        const int height = 375;//1988;
        const double focal_length = 4161.221;
        const double baseline = 176.252;
        const double cx1 = 1176.728;
        const double cy1 = 1011.728;
        const double cx2 = 1307.839;
        const double cy2 = 1011.728;
        const double doffs = cx2 - cx1;
        const int max_disparity = 60;
};

void scaleThenShow(Mat img) {
    Mat small;
    resize(img, small, Size(), 1, 1, CV_INTER_AREA);
    imshow(to_string(rand()), small);
}

int main() {
    Mat img1, img2, g1, g2;
    Mat disp, disp8;
    Mat out;

    img1 = imread("res/im0_mask.png");
    img2 = imread("res/im1_mask.png");

    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);
    out = g1.clone();
    cout << "depth " << out.depth() << endl
        << "channels " << out.channels() << endl;

    MyStereo my_stereo;
    my_stereo.drawDisparity(g1, g2, out);
    scaleThenShow(g1);
    scaleThenShow(out);
    waitKey(0);
    //    Ptr<StereoBM> sbm = StereoBM::create(18*16, 21);
    //    sbm->setMinDisparity(31);
    //    //sbm.state->SADWindowSize = 9;
    //    //sbm.state->numberOfDisparities = 112;
    //    //sbm.state->preFilterSize = 5;
    //    //sbm.state->preFilterCap = 61;
    //    //sbm.state->minDisparity = -39;
    //    //sbm.state->textureThreshold = 507;
    //    //sbm.state->uniquenessRatio = 0;
    //    //sbm.state->speckleWindowSize = 0;
    //    //sbm.state->speckleRange = 8;
    //    //sbm.state->disp12MaxDiff = 1;
    //
    //    sbm->compute(g1, g2, disp);
    //    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    //
    imwrite("OUTPUT.png", out);
    return 0;
}
