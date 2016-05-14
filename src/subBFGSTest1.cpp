

#include "subBFGSTest1.hpp"
#include <iostream>

subBFGSTest1::subBFGSTest1(double eps, int k_max, double h) : subBFGSBase(eps, k_max, h) {}

bool subBFGSTest1::init() {
    N_ = 2;
    B_ = MatrixXd::Identity(N_, N_);
    w_ = VectorXd::Zero(N_);
    w_[0] = 2;
    w_[1] = 1;
    return true;
}

double subBFGSTest1::ComputeObjective(const VectorXd& w) {
    if (w[0] >= fabs(w[1]))
        obj_ = 5 * sqrt(9 * w[0] * w[0] + 16 * w[1] * w[1]);
    else
        obj_ = 9 * w[0] + 16 * fabs(w[1]);
    return obj_;
}

bool subBFGSTest1::LineSearchStep() {
    eta_ = -(9 * w_[0] * p_[0] + 16 * p_[1] * w_[1]) / (9 * p_[0] * p_[0] + 16 * p_[1] * p_[1]);
    if ((w_[0] + eta_ * p_[0] >= fabs(w_[1] + eta_ * p_[1])) && (eta_ > 0))
        return true;
    else {
        if (81 * p_[0] * p_[0] - 256 * p_[1] * p_[1] < 0) {
            eta_ = -w_[1] / p_[1];
            return true;
//            if (eta_ > 0)
//                return true;
//            else {
//                std::cout << "Subdiff point is negative!" << std::endl;
//                return false;
        } else {
            std::cout << "Subdifferential point is not an optimum" << std::endl;
            if (9 * p_[0] + 16 * p_[1] < 0) {
                eta_ = 100;
                return true;
            } else {
                std::cout << "Gradient is always positive!" << std::endl;
                return false;
            }
        }
    }
}

bool subBFGSTest1::ComputeSubgradient(VectorXd* g) {
    double denominator = sqrt(9 * w_[0] * w_[0] + 16 * w_[1] * w_[1]);
    if (w_[0] >= fabs(w_[1])) {
        (*g)[0] = 45 * w_[0] / denominator;
        (*g)[1] = 80 * w_[1] / denominator;
    } else {
        (*g)[0] = 9;
        if (w_[1] > 0)
            (*g)[1] = 16;
        else if (w_[1] < 0)
            (*g)[1] = -16;
        else
            (*g)[1] = 16 * (2 * rand() / static_cast<double> (RAND_MAX) - 1);
    }
    return true;
}

bool subBFGSTest1::ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) {
    double denominator = sqrt(9 * w_[0] * w_[0] + 16 * w_[1] * w_[1]);
    if (w_[0] >= fabs(w_[1])) {
        (*g)[0] = 45 * w_[0] / denominator;
        (*g)[1] = 80 * w_[1] / denominator;
    } else {
        (*g)[0] = 9;
        if (p[1] > 0)
            (*g)[1] = 16;
        else
            (*g)[1] = -16;
    }
    return true;
}

subBFGSTest1::~subBFGSTest1() {}

