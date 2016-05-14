// Copyright 2015 Alexandr Katrutsa, INRIA

#include "subLBFGSBase.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

subLBFGSBase::subLBFGSBase(double epsilon, int k_max, double h) :
                        epsilon_(epsilon), k_max_(k_max), h_(h), N_(0), eta_(0),
                        obj_(0), num_iter_(0), step_tol_(0), MAX_NUM_ITER_(10000),
                        OBJ_TOL_(1e-9) {}

bool subLBFGSBase::solve() {
//    assert(step_tol_ > 0);
    assert(w_.rows() == N_);
    if (N_ < 1) {
        std::cout << "The dimension of the problem is less 1" << std::endl;
        return false;
    }
//    B_ = MatrixXd::Identity(N_, N_);
//    H_ = MatrixXd::Identity(N_, N_);
    VectorXd g = VectorXd::Zero(N_);
    if (!ComputeSubgradient(&g)) {
        std::cout << "Error in computing subgradient!" << std::endl;
        return false;
    }
//    std::cout << "First subgrad = \n" << g << std::endl;
    double new_objective = ComputeObjective(w_);
    double current_objective_tol = 1;
    while ((current_objective_tol > OBJ_TOL_) && (num_iter_ < MAX_NUM_ITER_)) {
        ++num_iter_;
        printf("Current iteration = %d\n", num_iter_);
        double current_obj = new_objective;
        if (!DescentDirection(g)) {
            return true;
        }
        std::cout << "Descent direction = \n" << p_ << std::endl;
//        if (!CheckDescentDirection()) {
//            return false;
//        }
//        printf("Descent direction is found!\n");
        if (!LineSearchStep()) {
            std::cout << "Error in computing step size!" << std::endl;
            return false;
        }
        if (eta_ == 0) {
            std::cout << "Step size is zero!" << std::endl;
//            if (!LineSearch()) {
//                return false;
//            }
            VectorXd subgrad = VectorXd::Zero(N_);
            ComputeSubgradient(&subgrad);
            if (subgrad.dot(subgrad) < EPS)
                return true;
            else {
                std::cout << "Try subgradient descent..." << std::endl;
                p_ = -subgrad;
                if (!CheckDescentDirection()) {
                    std::cout << "Subgradient is not descent direction!" << std::endl;
                    return true;
                }
                if (!LineSearchStep()) {
                    std::cout << "Error in computing step size!" << std::endl;
                    return false;
                }
                if (eta_ == 0) {
                    std::cout << "Step size is zero!" << std::endl;
                    return true;
                }
            }
        }
        VectorXd s = eta_ * p_;
//        if (s.array().abs().maxCoeff() < EPS) {
//            printf("The maximum absolute value in descent vector = %e\n", s.array().abs().maxCoeff());
//            std::cout << "The maximum absolute value of "
//                    "descent vector is too small!" << std::endl;
//            PrintInfSupSubgrad();
//            return true;
//        }
        if (!CheckStep()) {
            return false;
        }
        printf("Current step = %e\n", eta_);
//        if (!CheckWolfeConditions()) {
//            std::cout << "Violate Wolfe conditions!" << std::endl;
//            return false;
//        }
        w_ = w_ + s;
        VectorXd g2 = VectorXd::Zero(N_);
        if (!ComputeSubgradient(&g2)) {
            std::cout << "Error in computing subgradient!" << std::endl;
            return false;
        }
//        std::cout << "New subgrad = \n" << g2 << std::endl;
        VectorXd y = g2 - g;
        g = g2;
        double factor = 0;
        if (h_ - s.dot(y) / y.dot(y) > 0) {
            std::cout << "Secant equation fails!" << std::endl;
            printf("(s'y) / (y'y) = %e\n", s.dot(y) / y.dot(y));
            factor = h_ - s.dot(y) / y.dot(y);
        }
        s = s + factor * y;
        UpdateBH(s, y);
//        if (s.dot(y) > EPS) {
//            UpdateBH(s, y);
//            printf("Hessian and inverse hessian are updated!\n");
//        }
//        else
//            std::cout << "Secant equation fails! Skip update hessian and inverse hessian" << std::endl;
        new_objective = ComputeObjective(w_);
//        std::cout << "Current w = \n" << w_ << std::endl;
        double cur_diff = current_obj - new_objective;
        if (cur_diff < 0) {
            std::cout << "Objective increases on " << cur_diff << std::endl;
            std::cout << "The direction is ascent!" << std::endl;
            return false;
        }
        if ((current_obj < 0) && (new_objective < 0))
            current_objective_tol = (fabs(new_objective) - fabs(current_obj)) / fabs(new_objective);
        else
            current_objective_tol = cur_diff / current_obj;
        printf("(old_obj - new_obj) / old_obj = %e\n", current_objective_tol);
        printf("Current obj = %.5f\n", new_objective);
        std::cout << w_ << std::endl;
    }
    return true;
}

bool subLBFGSBase::LineSearch() {
    eta_ = 1e-2;
    while(!CheckWolfeConditions()) {
        eta_ *= 0.1;
    }
    return true;
}

void subLBFGSBase::UpdateBH(const VectorXd& s, const VectorXd& y) {
    history_s_.push_back(s);
    history_y_.push_back(y);
    double ro = 1 / y.dot(s);
    printf("Current ro = %e\n", ro);
    history_ro_.push_back(ro);
    history_b_.push_back(y / sqrt(y.dot(s)));
    VectorXd a = s;
    for (int  i = 0; i < history_a_.size(); ++i) {
        a = a + history_b_[i].dot(s) * history_b_[i] - history_a_[i].dot(s) * history_a_[i];
    }
    a = a / sqrt(s.dot(a));
    history_a_.push_back(a);
//    MatrixXd B = B_;
//    MatrixXd I = MatrixXd::Identity(N_, N_);
//    MatrixXd d1 = I - ro * s * y.transpose();
//    MatrixXd d2 = I - ro * y * s.transpose();
//    B_ = d1 * B_ * d2 + ro * s * s.transpose();
//    double yBy = y.dot(B_ * y);
//    VectorXd By = B_ * y;
//    MatrixXd Bys = By * s.transpose();
//    VectorXd yB = y.transpose() * B_;
//    MatrixXd syB = s * yB.transpose();
//    B_ = B_ - ro * (Bys  + syB) + (ro * ro * yBy + ro) * s * s.transpose();
//    std::cout << "Difference in matrices = " << (B - B_).array().abs().maxCoeff() << std::endl;
//    VectorXd Hs = H_ * s;
//    H_ = H_ + ro * y * y.transpose() - (Hs * Hs.transpose()) / (s.dot(Hs));
}

void subLBFGSBase::Restart() {
    history_y_.clear();
    history_s_.clear();
    history_ro_.clear();
    history_a_.clear();
    history_b_.clear();
}

bool subLBFGSBase::DescentDirection(const VectorXd& g) {
    VectorXd p = VectorXd::Zero(N_);
    MatVecInverseHessian(g, &p);
//    p = - B_ * g;
    p = -p;
    p_ = p;
//    std::vector<VectorXd> history_p;
//    history_p.push_back(p);
    VectorXd g_ = g;
    VectorXd g_asup = VectorXd::Zero(N_);
    if (!ArgSup(p, w_, &g_asup)) {
        std::cout << "Error in argsup computing!" << std::endl;
        return false;
    }
//    std::vector<double> pg_asup(1);
    double pg_asup = p.dot(g_asup);
//    std::vector<double> pg_(1);
    double pg_ = p.dot(g_);
    assert(pg_asup >= pg_);
    double min_M = pg_asup - 0.5 * pg_;
    double epsilon = min_M - 0.5 * pg_;
    int i = 0;
    std::vector<double> eps;
    while (((pg_asup > 0) || (epsilon > epsilon_)) &&
            (i < k_max_) && (epsilon > 0)) {
        eps.push_back(epsilon);
        VectorXd Bg = VectorXd::Zero(N_);
        MatVecInverseHessian(g_asup, &Bg);
//        double t1 = (g_ - g_asup).transpose() * Bg;
//        MatVecInverseHessian(g_ - g_asup, &Bg);
//        double t2 = (g_ - g_asup).transpose() * Bg;
//        double temp = t1 / t2;
        double nu_star = std::min(1.0, (pg_asup - pg_) / (2 * pg_asup - pg_ + g_asup.dot(Bg)));
        g_ = (1 - nu_star) * g_ + nu_star * g_asup;
//        MatVecInverseHessian(g_asup, &Bg);
        p = (1 - nu_star) * p - nu_star * Bg;
//        history_p.push_back(p);
        if (!ArgSup(p, w_, &g_asup)) {
            std::cout << "Error in argsup computing!" << std::endl;
            return false;
        }
//        pg_.push_back(p.dot(g_));
//        pg_asup.push_back(p.dot(g_asup));
        pg_ = p.dot(g_);
        pg_asup = p.dot(g_asup);
        double current_M = pg_asup - 0.5 * pg_;
        if (current_M < min_M) {
            min_M = current_M;
            p_ = p;
        }
        epsilon = min_M - 0.5 * pg_;
        ++i;
    }
//    double min_val = std::numeric_limits<double>::infinity();
//    size_t min_idx = 0;
//    for (size_t i = 0; i < history_p.size(); ++i) {
//        if (!ArgSup(history_p[i], w_, &g_asup)) {
//            std::cout << "Error in argsup computing!" << std::endl;
//            return false;
//        }
//        VectorXd Hp = VectorXd::Zero(N_);
//        MatVecHessian(history_p[i], &Hp);
//        double cur_val = 0.5 * history_p[i].transpose().dot(Hp) +
//                        g_asup.dot(history_p[i]);
//        if (cur_val < min_val) {
//            min_idx = i;
//        }
//    }
//    p_ = history_p[min_idx];
    
    if (pg_asup < 0.0 && i > 0)
        printf("Find a descent dir.: sup(gp) = %e in %d iterations with eps:%e\n", pg_asup, i, epsilon);
    else if (i > 0) {
        printf("No descent dir. in %d iterations; gap (%e) < %1.1e\n", i, epsilon, epsilon_);  
        return false;
    } else
        printf("Good direction: sup(gp) = %e\n", pg_asup);
//    if (g_asup.dot(p_) >= EPS)
//        return false;
    return true;
}

void subLBFGSBase::MatVecInverseHessian(const VectorXd& x, VectorXd* g) {
    *g = x;
    std::vector<double> alpha(history_s_.size(), 0);
    for (int i = history_s_.size() - 1; i >= 0; --i) {
        alpha[i] = history_ro_[i] * history_s_[i].dot(*g);
        *g = *g - alpha[i] * history_y_[i];
    }
//    MatrixXd H_0 = MatrixXd::Identity(N_, N_);
//    if (history_s.size() > 0) {
//        int last_idx = history_s.size() - 1;
//        H_0 = (history_s[last_idx].dot(history_y[last_idx]) /
//                history_y[last_idx].dot(history_y[last_idx])) * H_0;
//    }
    VectorXd r = (*g);
    for (int i = 0; i < history_s_.size(); ++i) {
        double beta = history_ro_[i] * history_y_[i].dot(r);
        r = r + history_s_[i] * (alpha[i] - beta);
    }
    *g = r;
}

void subLBFGSBase::MatVecHessian(const VectorXd& x, VectorXd* g) {
    (*g) = x;
    for (int i = 0; i < history_a_.size(); ++i) {
        (*g) = (*g) + history_b_[i] * history_b_[i].dot((*g)) - history_a_[i] * history_a_[i].dot((*g));
    }
}

double subLBFGSBase::CalculateEps(const std::vector<double>& pg_asup,
                                 const std::vector<double>& pg_, double pg) {
    double epsilon = pg_asup[0] - 0.5 * (pg_[0] + pg);
    for (size_t i = 1; i < pg_.size(); ++i) {
        double cur_eps = pg_asup[i] - 0.5 * (pg_[i] + pg);
        if (cur_eps < epsilon)
            epsilon = cur_eps;
    }
    return epsilon;
}

bool subLBFGSBase::CheckWolfeConditions() {
    double c1 = 1e-4;
    double c2 = 1 - c1;
    VectorXd g = VectorXd::Zero(N_);
    ArgSup(p_, w_, &g);
    VectorXd g_new = VectorXd::Zero(N_);
    ArgSup(p_, w_ + eta_ * p_, &g_new);
    // Check curvature
    if (g_new.dot(p_) < c2 * g.dot(p_)) {
        std::cout << "Error in curvature condition for step = " << eta_ << std::endl;
        printf("New dot product = %e\n", g_new.dot(p_));
        printf("c2 * old dot product = %e\n", c2 * g.dot(p_));
        return false;
    }
    double old_obj = ComputeObjective(w_);
    double new_obj = ComputeObjective(w_ + eta_ * p_);
    // Check sufficient decrease
    if (new_obj > old_obj + c1 * eta_ * g.dot(p_)) {
        printf("Error in sufficient decrease condition for step = %e\n", eta_);
        printf("New objective = %e\n", new_obj);
        printf("Old objective = %e\n", old_obj);
        printf("Old objective + supremum term  = %e\n",
                     old_obj + c1 * eta_ * g.dot(p_));
        return false;
    }
    return true;
}

bool subLBFGSBase::CheckDescentDirection() {
    VectorXd g = VectorXd::Zero(N_);
    ArgSup(p_, w_, &g);
    if (g.dot(p_) >= 0) {
        std::cout << "Found direction is not descent!" << std::endl;
        return false;
    }
    printf("Supremum g'p = %e\n", g.dot(p_));
    return true;
}

bool subLBFGSBase::CheckStep() {
    double old_obj = ComputeObjective(w_);
    double new_obj = ComputeObjective(w_ + eta_ * p_);
    if (old_obj <= new_obj) {
        std::cout << "New objective = " << new_obj << std::endl;
        std::cout << "Old objective = " << old_obj << std::endl;
        std::cout << "Step " << eta_ << " is not correct!" << std::endl;
        return false;
    }
    return true;
}

VectorXd subLBFGSBase::get_parameter() {
    return w_;
}

int subLBFGSBase::get_num_iter() {
    return num_iter_;
}

double subLBFGSBase::get_objective() {
    return ComputeObjective(w_);
}

void subLBFGSBase::PrintInfSupSubgrad() {}

subLBFGSBase::~subLBFGSBase() {}
