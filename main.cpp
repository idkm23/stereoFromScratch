#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <chrono>
#include <string>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MyStereoBM.h"

using namespace std;
using namespace cv;

void PrintVector(const vector<uchar>& vec) {
    cout << "{ ";
    for(auto i = vec.begin(); i != vec.end(); ++i) {
        cout << (int)*i << " ";
    }
    cout << "}\n";
}

void scaleThenShow(Mat img) {
    Mat small;
    resize(img, small, Size(), 1, 1, CV_INTER_AREA);
    imshow(to_string(rand()), small);
}

int main() {
    Mat img1, img2, g1, g2;
    Mat disp, disp8;

    img1 = imread("res/im0.png");
    img2 = imread("res/im1.png");

    cvtColor(img1, g1, CV_BGR2GRAY);
    cvtColor(img2, g2, CV_BGR2GRAY);

    MyStereoBM::State state;
    state.max_disparity = 70;
    state.window_size = 11;
    state.width = img1.cols;
    state.height = img1.rows;
    state.focal_length = 4161.221;
    state.baseline = 176.252;
    state.cx1 = 1176.728;
    state.cy1 = 1011.728;
    state.cx2 = 1307.839;
    state.cy2 = 1011.728;

    MyStereoBM my_stereo(state);

    using namespace std::chrono;
    milliseconds before = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch());
    Mat out = my_stereo.compute(g1, g2);
    milliseconds after = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch());
    std::chrono::duration<double, std::milli> dur = after - before;
    cout << "\nduration: " << dur.count()/1000.0 << endl;

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
