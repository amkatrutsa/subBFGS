// Copyright 2016 Alexandr Katrutsa, INRIA
#pragma once

#include <Eigen/Dense>
#include <vector>

typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;

class subBFGSBase {
    public:
        subBFGSBase(double epsilon, int k_max, double h);
        virtual ~subBFGSBase();
        bool solve();
        VectorXd get_parameter();
        int get_num_iter();
        double get_objective();
    protected:
        double epsilon_;    // see the paper
        int k_max_;     // see the paper
        double h_;  // see the paper
        int N_;     // dimension of the optimization problem
        double eta_;    // step size
        VectorXd w_;    // parameter vector
        VectorXd p_;    // descent direction
        MatrixXd B_;    // estimation of the inverse hessian
        double obj_;    // objective value
        int num_iter_;  // number of iteration after that converged
        const double EPS = 1e-20;  // tolerance of the non-zero subgradient
        const int MAX_NUM_ITER_;
        const double OBJ_TOL_;
        bool CheckWolfeConditions();
        bool CheckStep();
    private:
        virtual bool init() = 0;
        virtual bool LineSearchStep() = 0;
        virtual bool ComputeSubgradient(VectorXd* g) = 0;
        virtual double ComputeObjective(const VectorXd& w) = 0;
        virtual bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) = 0;
        virtual bool DescentDirection(const VectorXd& g);
        virtual void PrintInformationCurrentIter();
        bool LineSearch();
        subBFGSBase(const subBFGSBase& orig);
        subBFGSBase& operator = (const subBFGSBase& origin);
};

