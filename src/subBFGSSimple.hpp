// Copyright 2015 Alexandr Katrutsa, INRIA
#pragma once

#include "subLBFGSBase.hpp"

// This class implements the optimization of the function 
// f(x, y) = 10*|x - 5| + |y + 3|
// to test the subBFGS algorithm
class subBFGSSimple : public subLBFGSBase {
    public:
        subBFGSSimple(double epsilon, int k_max, double h);
        virtual ~subBFGSSimple();
        bool init();
    private:
        double tau_;
        double init_step_;
        bool LineSearchStep();
        bool ComputeSubgradient(VectorXd* g);
        bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g);
        double ComputeObjective(const VectorXd& w);
};
