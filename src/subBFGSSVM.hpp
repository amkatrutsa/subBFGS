// Copyright 2015 Alexandr Katrutsa, INRIA
#pragma once

#include "subLBFGSBase.hpp"

class subBFGSSVM : public subLBFGSBase {
    public:
        subBFGSSVM(double epsilon, int k_max, double h, const std::string& filename = "");
        virtual ~subBFGSSVM();
        bool init();
        double get_objective();
    private:
        MatrixXd X_;    // design matrix
        VectorXd y_;    // class labels
        double C_;  // misclassification cost
        std::string filename_;  // filename with data
        int M_;     // number of samples in train set
        bool ReadData();
        // This is a general function to compute step
        bool LineSearchStep();
        // This function recomputes subgradient in every subdifferential point
        bool LineSearchSubgradientRecomputation();
        // This function computes subgradient in the first subdifferential point
        // and then incrementally changes it without full re-computation
        bool LineSearchIncremental();
        bool ComputeSubgradient(VectorXd* g);
        bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g);
        double ComputeObjective(const VectorXd& w);
        // Save indices corresponding to the sorted eta vector in sorted_idx vector 
        // without sorting eta vector
        void ArgSort(const std::vector<double>& eta, std::vector<size_t>* sorted_idx);
        subBFGSSVM(const subBFGSSVM& orig);
        subBFGSSVM& operator = (const subBFGSSVM& orig);
};
