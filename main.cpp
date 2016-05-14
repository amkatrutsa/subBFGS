#include <cstdlib>
#include <iostream>
//#include "subBFGSSimple.hpp"
//#include "subBFGSSVM.hpp"
#include "subBFGSSVMBlocks.hpp"
//#include "subBFGSTest1.hpp"

int main(int argc, char** argv) {
    double eps = 1e-5;
    int iter = 3000;
    double h = 1e-8;
    Parameters p;
    if (!p.init(argc, argv)) {
        std::cout << "Error in parsing the command line argument!" << std::endl;
        return EXIT_FAILURE;
    }
//    std::string filename = "data.txt";
//    std::string block_list_filename = "../../../data/train_set8_bin/blocks.dat";
//    std::string block_list_filename = "../../../data/test_log/blocks_test_small.dat";
//    subBFGSSimple solver(eps, iter, h);
//    subBFGSSVM solver(eps, iter, h, filename);
//    subBFGSSVMBlocks solver(eps, iter, h, block_list_filename);
    subBFGSSVMBlocks solver(eps, iter, h, p);
//    subBFGSTest1 solver(eps, iter, h);
    if(!solver.init()) {
        std::cout << "Can not initialize problem" << std::endl;
        return EXIT_FAILURE;
    }
    if (!solver.solve()) {
        return EXIT_FAILURE;
    }
    std::cout << "The process converges in " << solver.get_num_iter() <<
                " iterations" << std::endl;
    if (!solver.write_to_file()) {
        std::cout << "Can not write the solution in the file!" << std::endl;
    }
    double objective = solver.get_objective();
    printf("The objective = %.6f\n", objective);
    printf("The final accuracy = %.6f\n", solver.compute_accuracy());
//    solver.check_solution("../../../data/test_log/param_trainset8_C3000.dat");
//    solver.check_solution("param_trainset8_C10.txt");
    return EXIT_SUCCESS;
}

