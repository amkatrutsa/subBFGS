// Copyright 2016 Alexandr Katrutsa, INRIA

#include "subBFGSBase.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

subBFGSBase::subBFGSBase(double epsilon, int k_max, double h) :
                        epsilon_(epsilon), k_max_(k_max), h_(h), N_(0), eta_(0),
                        obj_(0), num_iter_(0), MAX_NUM_ITER_(1000000),
                        OBJ_TOL_(1e-9) {}

bool subBFGSBase::solve() {
    assert(w_.rows() == N_);
    if (N_ < 1) {
        std::cout << "The dimension of the problem is less 1" << std::endl;
        return false;
    }
    double current_objective_tol = 1;
    double current_obj = ComputeObjective(w_);
    double new_objective = 0;
    VectorXd y = VectorXd::Zero(N_);
    VectorXd s = VectorXd::Zero(N_);
    while ((current_objective_tol > OBJ_TOL_) && (num_iter_ < MAX_NUM_ITER_)) {
        printf("Current iteration = %d\n", num_iter_ + 1);
        VectorXd g = VectorXd::Zero(N_);
        if (!ComputeSubgradient(&g)) {
            std::cout << "Error in computing subgradient!" << std::endl;
            return false;
        }
        if (num_iter_ == 0) {
            p_ = -g;
            if (!DescentDirection(g)) {
                return true;
            }
            if (!LineSearchStep()) {
                std::cout << "Error in computing step size!" << std::endl;
                return false;
            }
            if (!CheckStep()) {
                return false;
            }
            printf("Current step = %e\n", eta_);
        } else {
            y = g - y;
            double sy = s.dot(y);
            if (sy < 0) {
	      printf("Error: negative s'y %e\n", sy);
              return false;
            }
            if (fabs(sy) > 1e-100) {
                VectorXd By = B_ * y;
                s /= sy;
                B_ += (sy + y.dot(By)) * s * s.transpose() - (s * By.transpose() + By * s.transpose());
                p_ = -B_ * g;
                if (!DescentDirection(g)) {
                    return true;
                }
                if (!LineSearchStep()) {
                    std::cout << "Error in computing step size!" << std::endl;
                    return false;
                }
                if (!CheckStep()) {
                    return false;
                }
                printf("Current step = %e\n", eta_);
            } else {
                printf("Restart! eta = %e, s'y = %e\n", eta_, sy);
                break;
            }
        }
        s = eta_ * p_;
        w_ = w_ + s;
        new_objective = ComputeObjective(w_);
        y = g;
        ++num_iter_;
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
        PrintInformationCurrentIter();
        current_obj = new_objective;
    }
    return true;
}

bool subBFGSBase::LineSearch() {
    eta_ = 1e-2;
    while(!CheckWolfeConditions()) {
        eta_ *= 0.1;
    }
    return true;
}


bool subBFGSBase::DescentDirection(const VectorXd& g) {
    VectorXd p = p_;
    VectorXd g_ = g;
    VectorXd g_asup = VectorXd::Zero(N_);
    if (!ArgSup(p, w_, &g_asup)) {
        std::cout << "Error in argsup computing!" << std::endl;
        return false;
    }
    double pg_asup = p.dot(g_asup);
    double pg_ = p.dot(g_);
    assert(pg_asup >= pg_);
    double min_M = pg_asup - 0.5 * pg_;
    double epsilon = min_M - 0.5 * pg_;
    int i = 0;
    std::vector<double> eps;
    if (epsilon > epsilon_)
        printf("Start correcting procedure for non-smooth point...\n");
    while (((pg_asup > 0) || (epsilon > epsilon_)) &&
            (i < k_max_) && (epsilon > 0)) {
        if (i % 100 == 0)
            printf("Epsilon in iteration %d = %e > %e\n", i, epsilon, epsilon_);
        eps.push_back(epsilon);
        VectorXd Bg = B_ * g_asup;
        double nu_star = std::min(1.0, (pg_asup - pg_) / (2 * pg_asup - pg_ + g_asup.dot(Bg)));
        g_ = (1 - nu_star) * g_ + nu_star * g_asup;
        p = (1 - nu_star) * p - nu_star * Bg;
        if (!ArgSup(p, w_, &g_asup)) {
            std::cout << "Error in argsup computing!" << std::endl;
            return false;
        }
        pg_ = p.dot(g_);
        pg_asup = p.dot(g_asup);
        double current_M = pg_asup - 0.5 * pg_;
        if (current_M < min_M) {
            min_M = current_M;
            p_ = p;
        }
        epsilon = min_M - 0.5 * pg_;
        if (epsilon <= epsilon_)
            printf("Epsilon in iteration %d = %e <= %e\n", i + 1, epsilon, epsilon_);
        ++i;
    }
    if (pg_asup < 0.0 && i > 0) {
        if (i == k_max_)
            printf("Number of iteration is exceeded!\n");
        if (epsilon <= epsilon_)
            printf("Given tolerance is achieved!\n");
        printf("Find a descent dir.: sup(gp) = %e in %d iterations with eps = %e\n", pg_asup, i, epsilon);
    }
    else if (i > 0) {
        printf("No descent dir. in %d iterations; gap (%e) < %1.1e\n", i, epsilon, epsilon_);  
        return false;
    } else
        printf("Good direction: sup(gp) = %e\n", pg_asup);
    return true;
}

bool subBFGSBase::CheckWolfeConditions() {
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

bool subBFGSBase::CheckStep() {
    double old_obj = ComputeObjective(w_);
    double new_obj = ComputeObjective(w_ + eta_ * p_);
    if (old_obj < new_obj) {
        std::cout << "New objective = " << new_obj << std::endl;
        std::cout << "Old objective = " << old_obj << std::endl;
        std::cout << "Step " << eta_ << " is not correct!" << std::endl;
        return false;
    }
    return true;
}

VectorXd subBFGSBase::get_parameter() {
    return w_;
}

int subBFGSBase::get_num_iter() {
    return num_iter_;
}

double subBFGSBase::get_objective() {
    return ComputeObjective(w_);
}

void subBFGSBase::PrintInformationCurrentIter() {}

subBFGSBase::~subBFGSBase() {}
