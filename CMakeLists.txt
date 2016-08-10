CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(subBFGS)

# Allow me to specify compiler from cmake command line
if(NOT CMAKE_CXX_COMPILER)
    SET(CMAKE_CXX_COMPILER /usr/bin/clang++)
endif()
# SET(CMAKE_CXX_COMPILER /usr/bin/g++)

LIST(APPEND CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -g")

INCLUDE_DIRECTORIES(
./3rdparty/Eigen/
./src/
)

ADD_EXECUTABLE(subBFGS
./src/subLBFGSBase.cpp
./src/subBFGSBase.cpp
./src/subBFGSBase.hpp
./src/subBFGSTest1.cpp
./src/subBFGSSimple.cpp
./src/subBFGSSVM.cpp
./src/subBFGSSVMBlocks.cpp
./src/subBFGSBoxBoundBase.cpp
./src/Parameters.cpp
main.cpp
)
