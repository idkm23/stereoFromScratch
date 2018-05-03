to run on the motorcycle stereo pair:

    mkdir build/
    cd build
    cmake ..
    make
    ./main

to run on other stereo pairs:

    place the pair you want into build/res/ and name them im0.png and im1.png
    ./main

the pairs used for evaluation are found inside middlebury_evaluator/trainingQ

to measure the accuracy of a resulting depth map after a run, 
move the generated disp0MUNROE.pfm to the 
corresponding middlebury_evaluator/trainingQ/<the stereo pair> folder
and then run the middlebury_evaluator for that pair for the method MUNROE
(see the middlebury_evaluator README: http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3/README.txt)
