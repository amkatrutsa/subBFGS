// Copyright 2015 Alexander Katrutsa
#pragma once

#include <string>

// This structure stores the parameters parsed from the command line
struct Parameters {
    Parameters();
    bool init(int argc, char** argv);
    void exit_with_help();
    double C;
    // Tolerance of change the objective function:
    // if |Obj_old - Obj_current| / Obj_current < dwTol, stop training
//    double dwTol;
    std::string blocksFile;     // filename with list of block names
    std::string modelFile;      // filename to save the trained parameter vector
    // Filename to save data about training process
    std::string train_validate_data_filename;
//    int subSet;     // number of blocks used for training
};
