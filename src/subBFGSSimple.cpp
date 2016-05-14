// Copyright 2015 Alexandr Katrutsa, INRIA

#include "subBFGSSimple.hpp"
#include <iostream>

subBFGSSimple::subBFGSSimple(double epsilon, int k_max, double h) :
                subLBFGSBase(epsilon, k_max, h) {}

bool subBFGSSimple::init() {
    N_ = 2;
    tau_ = 0.8;
    init_step_ = 10;
    return true;
}

bool subBFGSSimple::ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) {
    if (w[0] > 5)
        (*g)[0] = 10;
    else if (w[0] < 5)
        (*g)[0] = -10;
    else {
        if (p[0] > 0)
            (*g)[0] = 10;
        else
            (*g)[0] = -10;
    }
    if (w[1] > -3)
        (*g)[1] = 1;
    else if (w[1] < -3)
        (*g)[1] = -1;
    else {
        if (p[1] > 0)
            (*g)[1] = 1;
        else
            (*g)[1] = -1;
    }
    return true;
}

double subBFGSSimple::ComputeObjective(const VectorXd& w) {
    obj_ = 10 * fabs(w(0) - 5) + fabs(w(1) + 3);
    return obj_;
}

bool subBFGSSimple::ComputeSubgradient(VectorXd* g) {
    if (w_[0] > 5)
        (*g)[0] = 10;
    else if (w_[0] < 5)
        (*g)[0] = -10;
    else
        (*g)[0] = 0;

    if (w_[1] > -3)
        (*g)[1] = 1;
    else if (w_[1] < -3)
        (*g)[1] = -1;
    else
        (*g)[1] = 0;
    return true;
}

bool subBFGSSimple::LineSearchStep() {
    VectorXd w = w_;
    double step = init_step_;
    double init_obj = ComputeObjective(w_);
    double cur_obj = ComputeObjective(w_ + step * p_);
    while (cur_obj >= init_obj) {
        step *= tau_;
        w = w_ + step * p_;
        cur_obj = ComputeObjective(w);
    }
    eta_ = step;
    return true;
}

subBFGSSimple::~subBFGSSimple() {}
