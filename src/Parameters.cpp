// Copyright 2015 Alexander Katrutsa

#include <iostream>
#include "Parameters.hpp"

Parameters::Parameters() : C(0), blocksFile(""), modelFile(""),
                           train_validate_data_filename("") {}

bool Parameters::init(int argc, char* argv[]) {
    if(argc <= 1) return false;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            std::cout << "The keys have to start with -" << std::endl;
            return false;
        }
        if (++i >= argc) {
            exit_with_help();
            return false;
        }
        switch (argv[i-1][1]) {
            case 'C':
                C = atof(argv[i]);
                break;

            case 'B':
                blocksFile = argv[i];
                break;

            case 'M':
                modelFile = argv[i];
                break;

            case 'b':
                train_validate_data_filename = argv[i];
                break;

            default:
                fprintf(stderr, "Unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
                return false;
        }
    }
    return true;
}

void Parameters::exit_with_help() {
    printf(
    "Usage: train [options]\n"
    "options:\n"
    "-f frame\n"
    "-r block file"
    "-Cp cost : set the parameter Cp (default 1)\n"
    "-Cm cost : set the parameter Cm (default 1)\n"
    "-T string : set the temporary progress save directory (f.e. \"tSave\\ \")\n"
    "-B string : set the file with blocks description\n"
    "-M string : set the output model file\n"
    "-t float : set the tolerance to deltaw\n"
    "-i int : set the maximum number of iterations\n"
    "-s : rosetta scoring\n"
    "-o int : polynomial order");
}
