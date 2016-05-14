// Copyright 2015 Alexandr Katrutsa, INRIA

#include "subBFGSBase.hpp"
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "Parameters.hpp"

struct sorted_blocks_idx {
    int total_eta_idx;
    size_t block_idx;
    int vector_idx;
};

// This class implements the block SVM learning algorithm. The parameter vector
// forms in the following way: the first num_block_ elements correspond to
// the intercept parameters b for each block, the next vector_dimension_ elements
// correspond to the normal vector to the separated hyperplane
class subBFGSSVMBlocks : public subBFGSBase {
    public:
        subBFGSSVMBlocks(double epsilon, int k_max, double h, const std::string& filename);
        subBFGSSVMBlocks(double epsilon, int k_max, double h, const Parameters& p);
        bool init();
        double get_objective();
        bool write_to_file();
        double compute_accuracy();
        bool check_solution(const std::string& solution_filename);
        ~subBFGSSVMBlocks();
    protected:
        std::string block_list_filename_;   // filename with list of blocks filenames
        std::vector<std::string> blocks_filenames_;
        std::string model_filename_;    // file to save the obtained parameter vector
        int vector_dimension_;  // dimension of the structured vectors from blocks, not the general parameter vector
        MatrixXd* X_;   // array with data matrix for every block
        MatrixXd X_natives_;    // matrix stores the native vectors for every block
        VectorXd* y_;   // array with class labels for every block
        VectorXd* C_;   // array with misclassification costs for every vector in every block
        double cost_;   // initial C in misclassification
        int total_vectors_;     // total number of vectors in all blocks
        size_t num_blocks_;
        const int NUM_ENTROPY_TERMS_ = 22;  // number of entropy terms in the parameter vector
        bool ReadBlocksList();
        bool ReadBinaryBlock(int block_idx);
    private:
        // This is a general function to compute step
        bool LineSearchStep();
        // This function computes subgradient in the first subdifferential point
        // and then incrementally changes it without full re-computation
        bool LineSearchIncremental();
        // This function computes the step like the function LineSearchIncremental(),
        // but additionally the result step does not violate
        // the inequality constraints if they are
//        bool LineSearchIncremSatisfiedIneqConstraints();
        bool ComputeSubgradient(VectorXd* g);
        bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g);
        double ComputeObjective(const VectorXd& w);
        // Save indices corresponding to the sorted eta vector in sorted_idx vector 
        // without sorting eta vector
        void ArgSort(const std::vector<double>& eta, std::vector<sorted_blocks_idx>* sorted_idx);
        void PrintInfSupSubgrad();
        void PrintInformationCurrentIter();
        subBFGSSVMBlocks(const subBFGSSVMBlocks& orig);
        subBFGSSVMBlocks& operator = (const subBFGSSVMBlocks& orig);
};

