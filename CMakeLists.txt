set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8)
project( stereoFromScratch )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( main main.cpp MyStereoBM.cpp )
target_link_libraries( main ${OpenCV_LIBS} )
FILE(COPY ${CMAKE_SOURCE_DIR}/res/ DESTINATION ./res/)
FILE(COPY ${CMAKE_SOURCE_DIR}/calib.txt DESTINATION .)
