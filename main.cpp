#include <cstdlib>
#include <iostream>
//#include "subBFGSSimple.hpp"
//#include "subBFGSSVM.hpp"
#include "subBFGSSVMBlocks.hpp"
//#include "subBFGSTest1.hpp"

int main(int argc, char** argv) {
    double eps = 1e-4;
    int iter = 1000000;
    double h = 1e-8;
    Parameters p;
    if (!p.init(argc, argv)) {
        std::cout << "Error in parsing the command line argument!" << std::endl;
        return EXIT_FAILURE;
    }
    subBFGSSVMBlocks solver(eps, iter, h, p);
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
    printf("The objective = %e\n", objective);
    printf("The final accuracy = %e\n", solver.compute_accuracy());
//    solver.check_solution("../../../data/test_log/param_trainset8_C3000.dat");
//    solver.check_solution("param_trainset8_C10.txt");
    return EXIT_SUCCESS;
}

