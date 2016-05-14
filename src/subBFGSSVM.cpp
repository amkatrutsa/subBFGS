// Copyright 2015 Alexandr Katrutsa, INRIA

#include "subBFGSSVM.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

subBFGSSVM::subBFGSSVM(double epsilon, int k_max, double h, const std::string& filename) :
                subLBFGSBase(epsilon, k_max, h), filename_(filename) {}

bool subBFGSSVM::init() {
    if (!ReadData()) {
        std::cerr << "Can't read file " << filename_ << std::endl;
        return false;
    }
    C_ = 100;
    step_tol_ = 1e-7 / C_;
    return true;
}

bool subBFGSSVM::ReadData() {
    std::ifstream myfile(filename_);
    if (!myfile.good()) {
        std::cerr << "Can't open file " << filename_ << std::endl;
        return false;
    }
    myfile >> M_ >> N_;
    X_.resize(M_, N_);
    y_.resize(M_);
    for (int  i = 0; i < M_; ++i) {
        myfile >> y_[i];
        for (int j = 0; j < N_; ++j) {
            myfile >> X_(i, j);
        }
    }
    return true;
}

bool subBFGSSVM::ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) {
    assert(g->rows() == N_);
    assert(p.rows() == N_);
    assert(C_ > 0);
    g->segment(1, N_ - 1) = w.segment(1, N_ - 1);
    (*g)[0] = 0;
    VectorXd loss = 1 - y_.cwiseProduct(X_ * w).array();
    for (int i = 0; i < M_; ++i) {
        if (loss[i] > 0) {
            g->segment(1, N_ - 1) -= C_ * y_[i] * X_.row(i).segment(1, N_ - 1);
            (*g)[0] -= C_ * y_[i];
        } else if (loss[i] == 0) {
            if (y_[i] * X_.row(i).segment(1, N_ - 1).dot(p.segment(1, N_ - 1)) < 0)
                g->segment(1, N_ - 1) -= C_ * y_[i] * X_.row(i).segment(1, N_ - 1);
            if (p[0] * y_[i] < 0)
                (*g)[0] -= C_ * y_[i];
        }
    }
    return true;
}

double subBFGSSVM::ComputeObjective(const VectorXd& w) {
    assert(C_ > 0);
    obj_ = 0.5 * w.segment(1, N_ - 1).dot(w.segment(1, N_ - 1));
    VectorXd loss = 1 - y_.cwiseProduct(X_ * w).array();
    for (int i = 0; i < loss.rows(); ++i) {
        if (loss[i] > 0) {
            obj_ += C_ * loss[i];
        }
    }
    return obj_;
}

bool subBFGSSVM::ComputeSubgradient(VectorXd* g) {
    assert(C_ > 0);
    assert(g->rows() == N_);
    g->segment(1, N_ - 1) = w_.segment(1, N_ - 1);
    VectorXd loss = 1 - y_.cwiseProduct(X_ * w_).array();
    for (int i = 0; i < loss.rows(); ++i) {
        if (loss[i] > 0) {
            (*g)[0] -= C_ * y_[i];
            g->segment(1, N_ - 1) -= C_ * y_[i] * X_.row(i).segment(1, N_ - 1);
        }
    }
    return true;
}

bool subBFGSSVM::LineSearchStep() {
    if (!LineSearchIncremental()) {
//    if (!LineSearchSubgradientRecomputation()) {
        std::cout << "Can not find optimal step!" << std::endl;
        return false;
    }
    return true;
}

bool subBFGSSVM::LineSearchSubgradientRecomputation() {
    assert(C_ > 0);
    double norm_p2 = p_.segment(1, N_ - 1).dot(p_.segment(1, N_ - 1));
    double wp =  w_.segment(1, N_ - 1).dot(p_.segment(1, N_ - 1));
    VectorXd f = y_.cwiseProduct(X_ * w_);
    VectorXd delta_f = y_.cwiseProduct(X_ * p_);
    std::vector<double> subdif_eta(M_, 0);
    for (int i = 0; i < M_; ++i) {
        if (delta_f(i) != 0)
            subdif_eta[i] = (1 - f(i)) / delta_f(i);
    }
    std::vector<size_t> init_idx;
    std::vector<double> eta;
    for (size_t i = 0; i < subdif_eta.size(); ++i) {
        if (subdif_eta[i] > 0) {
            eta.push_back(subdif_eta[i]);
            init_idx.push_back(i);
        }
    }
    std::vector<size_t> sorted_idx(eta.size(), 0);
    ArgSort(eta, &sorted_idx);
    double init_grad = wp + eta[sorted_idx[0]] * norm_p2;
    for (int i = 0; i < f.rows(); ++i) {
        if (f[i] + eta[sorted_idx[0]] * delta_f[i] < 1) {
            init_grad -= C_ * delta_f[i];
        }
    }
    double init_grad_sup = 0;
    double init_grad_inf = 0;
    if (delta_f[init_idx[sorted_idx[0]]] < 0) {
        init_grad_sup = init_grad - C_ * delta_f[init_idx[sorted_idx[0]]];
        init_grad_inf = init_grad;
    } else {
        init_grad_sup = init_grad;
        init_grad_inf = init_grad - C_ * delta_f[init_idx[sorted_idx[0]]];
    }
//    std::cout << "Left grad in " << eta[sorted_idx[0]] << " = " << init_grad_inf << std::endl;
//    std::cout << "Right grad in " << eta[sorted_idx[0]] << " = " << init_grad_sup << std::endl;
    // Check supremum and infimum of the subgradient in the first subdifferential point
    // If infimum and supremum absolute values are small enough then current
    // vector w_ is already optimal
    if ((fabs(init_grad_inf) < EPS) && (fabs(init_grad_sup) < EPS)) {
        eta_ = 0;
        return true;
    }
    // If 0 lies between infimum and supremum, then this subdifferential point is
    // an optimum step
    if ((init_grad_inf < 0) && (init_grad_sup > 0)) {
        eta_ = eta[sorted_idx[0]];
        return true;
    }
    // If both infimum and supremum are positive, then the optimum step is
    // between 0 and minimum subdifferential point or in the border of this segment 
    if ((init_grad_inf > 0) && (init_grad_sup > 0)) {
        double test_eta = eta[sorted_idx[0]] * 0.5;
        eta_ = -wp;
        for (int i = 0; i < f.rows(); ++i) {
            if (f[i] + test_eta * delta_f[i] < 1)
                eta_ += C_ * delta_f[i]; 
        }
        eta_ /= norm_p2;
        double left_obj = ComputeObjective(w_);
        double right_obj = ComputeObjective(w_ + eta[sorted_idx[0]] * p_);
        // If eta_ lies between 0 and eta[sorted_idx[0]], check where is 
        // the objective smaller: in 0, in eta_ or in eta[sorted_idx[0]]
        if ((eta_ >= 0) && (eta_ <= eta[sorted_idx[0]])) {
            double middle_obj = ComputeObjective(w_ + eta_ * p_);
            if ((left_obj < right_obj) && (left_obj < middle_obj)) {
                eta_ = 0;
                return true;
            }
            if ((middle_obj < left_obj) && (middle_obj < right_obj))
                return true;
            if ((right_obj < left_obj) && (right_obj < middle_obj)) {
                eta_ = eta[sorted_idx[0]];
                return true;
            }
        } else {
            if (right_obj < left_obj)
                eta_ = eta[sorted_idx[0]];
            else
                eta_ = 0;
            return true;
        }
    }
    // Loop over all subdifferential points
    for (size_t i = 1; i < eta.size(); ++i) {
        double cur_grad = wp + eta[sorted_idx[i]] * norm_p2;
        double cur_grad_sup = 0;
        double cur_grad_inf = 0;
        // Re-compute subgradient in every subdifferential points
        for (int j = 0; j < f.rows(); ++j) {
            if (f[j] + eta[sorted_idx[i]] * delta_f[j] < 1)
                cur_grad -= C_ * delta_f[j];
        }
        if (delta_f[init_idx[sorted_idx[i]]] < 0) {
            cur_grad_sup = cur_grad - C_ * delta_f[init_idx[sorted_idx[i]]];
            cur_grad_inf = cur_grad;
        } else {
            cur_grad_sup = cur_grad;
            cur_grad_inf = cur_grad - C_ * delta_f[init_idx[sorted_idx[i]]];
        }
//        std::cout << "Left grad in " << eta[sorted_idx[i]] << " = " << cur_grad_inf << std::endl;
//        std::cout << "Right grad in " << eta[sorted_idx[i]] << " = " << cur_grad_sup << std::endl;
        // Analyse the supremum and infimum of the current subdifferentiabe point
        if ((cur_grad_inf < 0) && (cur_grad_sup > 0)) {
            eta_ = eta[sorted_idx[i]];
            return true;
        }
        if ((init_grad_sup < 0) && (cur_grad_inf > 0)) {
            double test_eta = 0.5 * (eta[sorted_idx[i-1]] + eta[sorted_idx[i]]);
            eta_ = -wp;
            for (int i = 0; i < f.rows(); ++i) {
                if (f[i] + test_eta * delta_f[i] < 1)
                    eta_ += C_ * delta_f[i];
            }
            eta_ /= norm_p2;
            double left_obj = ComputeObjective(w_ + eta[sorted_idx[i-1]] * p_);
            double right_obj = ComputeObjective(w_ + eta[sorted_idx[i]] * p_);
            if ((eta_ <= eta[sorted_idx[i]]) && (eta_ >= eta[sorted_idx[i-1]])) {
                double middle_obj = ComputeObjective(w_ + eta_ * p_);
                if ((left_obj < right_obj) && (left_obj < middle_obj)) {
                    eta_ = eta[sorted_idx[i-1]];
                    return true;
                }
                if ((middle_obj < left_obj) && (middle_obj < right_obj))
                    return true;
                if ((right_obj < left_obj) && (right_obj < middle_obj)) {
                    eta_ = eta[sorted_idx[i]];
                    return true;
                }
            }
            else {
                if (left_obj < right_obj) {
                    eta_ = eta[sorted_idx[i-1]];
                    return true;
                } else {
                    eta_ = eta[sorted_idx[i]];
                    return true;
                }
            }
        }
        init_grad_sup = cur_grad_sup;
    }
    return false;
}

bool subBFGSSVM::LineSearchIncremental() {
    assert(C_ > 0);
    double norm_p2 = p_.segment(1, N_ - 1).dot(p_.segment(1, N_ - 1));
    double wp =  w_.segment(1, N_ - 1).dot(p_.segment(1, N_ - 1));
    VectorXd f = y_.cwiseProduct(X_ * w_);
    VectorXd delta_f = y_.cwiseProduct(X_ * p_);
    std::vector<double> subdif_eta(M_, 0);
    for (int i = 0; i < M_; ++i) {
        if (delta_f(i) != 0)
            subdif_eta[i] = (1 - f(i)) / delta_f(i);
    }
    std::vector<size_t> init_idx;
    std::vector<double> eta;
    for (size_t i = 0; i < subdif_eta.size(); ++i) {
        if (subdif_eta[i] > 0) {
            eta.push_back(subdif_eta[i]);
            init_idx.push_back(i);
        }
    }
    std::vector<size_t> sorted_idx(eta.size(), 0);
    ArgSort(eta, &sorted_idx);
    double init_grad_norm_w = wp + eta[sorted_idx[0]] * norm_p2;
    double init_grad_misclass = 0;
    for (int i = 0; i < f.rows(); ++i) {
        if (f[i] + eta[sorted_idx[0]] * delta_f[i] <= 1) {
            init_grad_misclass -= C_ * delta_f[i];
        }
    }
    double init_grad = init_grad_norm_w + init_grad_misclass;
    double init_grad_sup = 0;
    double init_grad_inf = 0;
    if (delta_f[init_idx[sorted_idx[0]]] < 0) {
        init_grad_sup = init_grad - C_ * delta_f[init_idx[sorted_idx[0]]];
        init_grad_inf = init_grad;
    } else {
        init_grad_sup = init_grad + C_ * delta_f[init_idx[sorted_idx[0]]];
        init_grad_inf = init_grad;
    }
    // Check supremum and infimum of the subgradient in the first subdifferential point
    // If infimum and supremum absolute values are small enough then current
    // vector w_ is already optimal
    if ((fabs(init_grad_inf) < EPS) && (fabs(init_grad_sup) < EPS)) {
        eta_ = 0;
        return true;
    }
    // If 0 lies between infimum and supremum, then this subdifferential point is
    // an optimum step
    if ((init_grad_inf < 0) && (init_grad_sup > 0)) {
        eta_ = eta[sorted_idx[0]];
        return true;
    }
    // If both infimum and supremum are positive, then the optimum step is
    // between 0 and minimum subdifferential point or in the border of this segment 
    if ((init_grad_inf > 0) && (init_grad_sup > 0)) {
        double test_eta = eta[sorted_idx[0]] * 0.5;
        eta_ = -wp;
        for (int i = 0; i < f.rows(); ++i) {
            if (f[i] + test_eta * delta_f[i] < 1)
                eta_ += C_ * delta_f[i]; 
        }
        eta_ /= norm_p2;
        double left_obj = ComputeObjective(w_);
        double right_obj = ComputeObjective(w_ + eta[sorted_idx[0]] * p_);
        // If eta_ lies between 0 and eta[sorted_idx[0]], check where is 
        // the objective smaller: in 0, in eta_ or in eta[sorted_idx[0]]
        if ((eta_ >= 0) && (eta_ <= eta[sorted_idx[0]])) {
            double middle_obj = ComputeObjective(w_ + eta_ * p_);
            if ((left_obj < right_obj) && (left_obj < middle_obj)) {
                eta_ = 0;
                return true;
            }
            if ((middle_obj < left_obj) && (middle_obj < right_obj))
                return true;
            if ((right_obj < left_obj) && (right_obj < middle_obj)) {
                eta_ = eta[sorted_idx[0]];
                return true;
            }
        } else {
            if (right_obj < left_obj)
                eta_ = eta[sorted_idx[0]];
            else
                eta_ = 0;
            return true;
        }
    }
    double cur_grad_norm_w = init_grad_norm_w;
    double cur_grad_misclass = init_grad_sup - init_grad_norm_w;
    double cur_grad_inf = 0;
    double cur_grad_sup = 0;
    // Loop over all subdifferential points
    for (size_t i = 1; i < eta.size(); ++i) {
        // Incremental change the subgradient in the current subdifferential point
        cur_grad_norm_w += (eta[sorted_idx[i]] - eta[sorted_idx[i-1]]) * norm_p2;
        double cur_grad = cur_grad_norm_w + cur_grad_misclass;
        if (delta_f[init_idx[sorted_idx[i]]] < 0) {
            cur_grad_sup = cur_grad - C_ * delta_f[init_idx[sorted_idx[i]]];
            cur_grad_inf = cur_grad;
        } else {
            cur_grad_sup = cur_grad + C_ * delta_f[init_idx[sorted_idx[i]]];
            cur_grad_inf = cur_grad;
        }
//        std::cout << "Left grad in " << eta[sorted_idx[i]] << " = " << cur_grad_inf << std::endl;
//        std::cout << "Right grad in " << eta[sorted_idx[i]] << " = " << cur_grad_sup << std::endl;
        // Analyse the supremum and infimum of the current subdifferentiabe point
        if ((cur_grad_inf < 0) && (cur_grad_sup > 0)) {
            eta_ = eta[sorted_idx[i]];
            return true;
        }
        if ((init_grad_sup < 0) && (cur_grad_inf > 0)) {
            double test_eta = 0.5 * (eta[sorted_idx[i-1]] + eta[sorted_idx[i]]);
            eta_ = -wp;
            for (int i = 0; i < f.rows(); ++i) {
                if (f[i] + test_eta * delta_f[i] < 1)
                    eta_ += C_ * delta_f[i];
            }
            eta_ /= norm_p2;
            double left_obj = ComputeObjective(w_ + eta[sorted_idx[i-1]] * p_);
            double right_obj = ComputeObjective(w_ + eta[sorted_idx[i]] * p_);
            if ((eta_ <= eta[sorted_idx[i]]) && (eta_ >= eta[sorted_idx[i-1]])) {
                double middle_obj = ComputeObjective(w_ + eta_ * p_);
                if ((left_obj < right_obj) && (left_obj < middle_obj)) {
                    eta_ = eta[sorted_idx[i-1]];
                    return true;
                }
                if ((middle_obj < left_obj) && (middle_obj < right_obj))
                    return true;
                if ((right_obj < left_obj) && (right_obj < middle_obj)) {
                    eta_ = eta[sorted_idx[i]];
                    return true;
                }
            }
            else {
                if (left_obj < right_obj) {
                    eta_ = eta[sorted_idx[i-1]];
                    return true;
                } else {
                    eta_ = eta[sorted_idx[i]];
                    return true;
                }
            }
        }
//        init_grad_inf = cur_grad_inf;
        init_grad_sup = cur_grad_sup;
        cur_grad_misclass = cur_grad_sup - cur_grad_norm_w;
    }
    std::cout << "Can not find optimal step!" << std::endl;
    return false;
}

void subBFGSSVM::ArgSort(const std::vector<double>& eta, std::vector<size_t>* sorted_idx) {
    assert(sorted_idx->size() == eta.size());
    for (size_t i = 0; i < sorted_idx->size(); ++i)
        (*sorted_idx)[i] = i;
    std::sort(sorted_idx->begin(), sorted_idx->end(), 
             [&eta] (size_t i1, size_t i2) { return eta[i1] < eta[i2]; });
}

double subBFGSSVM::get_objective() {
    printf("||w||^2 / 2 = %.6f\n", 0.5 * w_.segment(1, N_ - 1).dot(w_.segment(1, N_ - 1)));
    double misclass_error = 0;
    VectorXd loss = 1 - y_.cwiseProduct(X_ * w_).array();
    for (int i = 0; i < loss.rows(); ++i) {
        if (loss[i] > 0)
            misclass_error += C_ * loss[i];
    }
    printf("Misclassification error = %.6f\n", misclass_error);
    return ComputeObjective(w_);
} 

subBFGSSVM::~subBFGSSVM() {}
