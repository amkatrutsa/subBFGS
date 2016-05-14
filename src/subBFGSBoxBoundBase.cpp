// Copyright 2016 Alexandr Katrutsa, INRIA

#include "subBFGSBoxBoundBase.hpp"
#include <iostream>

subBFGSBoxBoundBase::subBFGSBoxBoundBase(double epsilon, int k_max, double h) :
                        epsilon_(epsilon), k_max_(k_max), h_(h), N_(0), eta_(0),
                        obj_(0), num_iter_(0), MAX_NUM_ITER_(10000),
                        OBJ_TOL_(1e-9) {}

bool subBFGSBoxBoundBase::solve() {
    if (N_ < 1) {
        std::cout << "The dimension of the problem is less 1" << std::endl;
        return false;
    }
    w_ = VectorXd::Zero(N_);
    VectorXd g = VectorXd::Zero(N_);
    if (!ComputeSubgradient(&g)) {
        std::cout << "Error in computing subgradient!" << std::endl;
        return false;
    }
    double new_objective = ComputeObjective(w_);
    double obj_tol = 1;
    while ((num_iter_ < MAX_NUM_ITER_) && (obj_tol > OBJ_TOL_)) {
        ++num_iter_;
        printf("Current iteration = %d\n", num_iter_);
        double current_obj = new_objective;
        if (!DescentDirection(g)) {
            return true;
        }
        if (!CheckDescentDirection()) {
            return false;
        }
        printf("Descent direction is found!\n");
        if (!DetectBorderPoints()) {
            std::cout << "Error in border point detection!" << std::endl;
            return false;
        }
        if (!LineSearchStep()) {
            std::cout << "Error in computing step size!" << std::endl;
            return false;
        }
        VectorXd s = eta_ * p_;
        if (!CheckStep()) {
            return false;
        }
        printf("Current step = %e\n", eta_);
        if (!CheckWolfeConditions()) {
            std::cout << "Violate Wolfe conditions!" << std::endl;
            return false;
        }
        w_ = w_ + s;
        VectorXd g2 = VectorXd::Zero(N_);
        if (!ComputeSubgradient(&g2)) {
            std::cout << "Error in computing subgradient!" << std::endl;
            return false;
        }
        VectorXd y = g2 - g;
        g = g2;
        s = s + std::max(0.0, h_ - s.dot(y) / y.dot(y)) * y;
        UpdateBH(s, y);
        printf("Hessian and inverse hessian are updated!\n");
        new_objective = ComputeObjective(w_);
        double cur_diff = current_obj - new_objective;
        if (cur_diff < 0) {
            std::cout << "Objective increases on " << cur_diff << std::endl;
            std::cout << "The direction becomes ascent!" << std::endl;
            return false;
        }
        obj_tol = cur_diff / current_obj;
        printf("(old_obj - new_obj) / old_obj = %e\n", obj_tol);
        printf("Current obj = %.5f\n", new_objective);
    }
    return false;
}

bool subBFGSBoxBoundBase::DescentDirection(const VectorXd& g) {
    VectorXd p = VectorXd::Zero(N_);
    MatVecInverseHessian(g, &p);
//    p = - B_ * g;
    p = -p;
    std::vector<VectorXd> history_p;
    history_p.push_back(p);
    VectorXd g_ = g;
    VectorXd g_asup = VectorXd::Zero(N_);
    if (!ArgSup(p, w_, &g_asup)) {
        std::cout << "Error in argsup computing!" << std::endl;
        return false;
    }
    std::vector<double> pg_asup(1);
    pg_asup[0] = p.dot(g_asup);
    std::vector<double> pg_(1);
    pg_[0] = p.dot(g_);
    double epsilon = pg_asup[0] - pg_[0];
    int i = 0;
    while (((g_asup.dot(p) > 0) || (epsilon > epsilon_)) &&
            (i < k_max_) && (epsilon > 0)) {
        VectorXd Bg = VectorXd::Zero(N_);
        MatVecInverseHessian(g_, &Bg);
        double t1 = (g_ - g_asup).transpose() * Bg;
        MatVecInverseHessian(g_ - g_asup, &Bg);
        double t2 = (g_ - g_asup).transpose() * Bg;
        double temp = t1 / t2;
        double nu_star = std::min(1.0, temp);
        g_ = (1 - nu_star) * g_ + nu_star * g_asup;
        MatVecInverseHessian(g_asup, &Bg);
        p = (1 - nu_star) * p - nu_star * Bg;
        history_p.push_back(p);
        if (!ArgSup(p, w_, &g_asup)) {
            std::cout << "Error in argsup computing!" << std::endl;
            return false;
        }
        pg_.push_back(p.dot(g_));
        pg_asup.push_back(p.dot(g_asup));
        double pg = p.dot(g_);
        epsilon = CalculateEps(pg_asup, pg_, pg);
        ++i;
    }
    double min_val = std::numeric_limits<double>::infinity();
    size_t min_idx = 0;
    for (size_t i = 0; i < history_p.size(); ++i) {
        if (!ArgSup(history_p[i], w_, &g_asup)) {
            std::cout << "Error in argsup computing!" << std::endl;
            return false;
        }
        VectorXd Hp = VectorXd::Zero(N_);
        MatVecHessian(history_p[i], &Hp);
        double cur_val = 0.5 * history_p[i].transpose().dot(Hp) +
                        g_asup.dot(history_p[i]);
        if (cur_val < min_val) {
            min_idx = i;
        }
    }
    p_ = history_p[min_idx];
    if (g_asup.dot(p_) >= EPS)
        return false;
    return true;
}

void subBFGSBoxBoundBase::MatVecInverseHessian(const VectorXd& x, VectorXd* g) {
    *g = x;
    std::vector<double> alpha(history_s.size(), 0);
    for (int i = history_s.size() - 1; i >= 0; --i) {
        alpha[i] = history_ro[i] * history_s[i].dot(*g);
        *g = *g - alpha[i] * history_y[i];
    }
    VectorXd r = (*g);
    for (int i = 0; i < history_s.size(); ++i) {
        double beta = history_ro[i] * history_y[i].dot(r);
        r = r + history_s[i] * (alpha[i] - beta);
    }
    *g = r;
}

void subBFGSBoxBoundBase::MatVecHessian(const VectorXd& x, VectorXd* g) {
    (*g) = x;
    for (int i = 0; i < history_a.size(); ++i) {
        (*g) = (*g) + history_b[i] * history_b[i].dot((*g)) - history_a[i] * history_a[i].dot((*g));
    }
}

double subBFGSBoxBoundBase::CalculateEps(const std::vector<double>& pg_asup,
                                 const std::vector<double>& pg_, double pg) {
    double epsilon = pg_asup[0] - 0.5 * (pg_[0] + pg);
    for (size_t i = 1; i < pg_.size(); ++i) {
        double cur_eps = pg_asup[i] - 0.5 * (pg_[i] + pg);
        if (cur_eps < epsilon)
            epsilon = cur_eps;
    }
    return epsilon;
}

bool subBFGSBoxBoundBase::CheckWolfeConditions() {
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
        std::cout << "Error in sufficient decrease condition for step = " << eta_ << std::endl;
        printf("New objective = %e\n", new_obj);
        printf("Old objective = %e\n", old_obj);
        printf("Old objective + supremum term  = %e",
                     old_obj + c1 * eta_ * g.dot(p_));
        return false;
    }
    return true;
}

bool subBFGSBoxBoundBase::CheckDescentDirection() {
    VectorXd g = VectorXd::Zero(N_);
    ArgSup(p_, w_, &g);
    if (g.dot(p_) >= 0) {
        std::cout << "Found direction is not descent!" << std::endl;
        return false;
    }
    printf("Supremum g'p = %e\n", g.dot(p_));
    return true;
}

void subBFGSBoxBoundBase::UpdateBH(const VectorXd& s, const VectorXd& y) {
    history_s.push_back(s);
    history_y.push_back(y);
    double ro = 1 / y.dot(s);
    printf("Current ro = %e\n", ro);
    history_ro.push_back(ro);
    history_b.push_back(y / sqrt(y.dot(s)));
    VectorXd a = s;
    for (int  i = 0; i < history_a.size(); ++i) {
        a = a + history_b[i].dot(s) * history_b[i] - history_a[i].dot(s) * history_a[i];
    }
    a = a / sqrt(s.dot(a));
    history_a.push_back(a);
}

bool subBFGSBoxBoundBase::CheckStep() {
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

VectorXd subBFGSBoxBoundBase::get_parameter() {
    return w_;
}

int subBFGSBoxBoundBase::get_num_iter() {
    return num_iter_;
}
subBFGSBoxBoundBase::~subBFGSBoxBoundBase() {}

