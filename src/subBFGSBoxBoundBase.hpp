// Copyright 2016 Alexandr Katrutsa, INRIA
#pragma once

#include <Eigen/Dense>
#include <vector>

typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;

class subBFGSBoxBoundBase {
    public:
        subBFGSBoxBoundBase(double eps, int k_max, double h);
        virtual ~subBFGSBoxBoundBase();
        bool solve();
        VectorXd get_parameter();
        int get_num_iter();
    protected:
        double epsilon_;    // see the paper
        int k_max_;     // see the paper
        double h_;  // see the paper
        int N_;     // dimension of the optimization problem
        double eta_;    // step size
        VectorXd w_;    // parameter vector
        VectorXd p_;    // descent direction
        double obj_;    // objective value
        int num_iter_;  // number of iteration after that converged
        const double EPS = 1e-20;  // tolerance of the non-zero element
        const int MAX_NUM_ITER_;
        const double OBJ_TOL_;
        // These vectors store the history of y, s and ro
        std::vector<VectorXd> history_y;
        std::vector<VectorXd> history_s;
        std::vector<double> history_ro;
        // These vectors store some specific vectors to fast multiply hessian by vector
        // More details see in 
        // http://users.iems.northwestern.edu/~nocedal/PDFfiles/representations.pdf Section 4.2
        std::vector<VectorXd> history_a;
        std::vector<VectorXd> history_b;
        MatrixXd B_reduced_;
        MatrixXd H_reduced_;
        VectorXd p_reduced_;
    private:
        virtual bool init() = 0;
        virtual double ComputeObjective(const VectorXd& w) = 0;
        virtual bool ComputeSubgradient(VectorXd* g) = 0;
        virtual bool LineSearchStep() = 0;
        virtual bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) = 0;
        virtual bool DetectBorderPoints() = 0;
        virtual bool DescentDirection(const VectorXd& g);
        void UpdateBH(const VectorXd& s, const VectorXd& y);
        double CalculateEps(const std::vector<double>& pg_asup,
                            const std::vector<double>& pg_, double pg);
        bool CheckWolfeConditions();
        bool CheckDescentDirection();
        bool CheckStep();
        // This method implements the multiplication of the hessian estimation on
        // the given vector x. Instead of explicit storing the hessian estimation,
        // we store the history of differences between w from two sequential iterations and
        // between gradients of the objective in the points from two sequential iterations and
        // compute product incrementally.
        // More details, see in
        // J. Nocedal, S. Wright Numerical Optimization, Springer, 2006
        // URL: http://home.agh.edu.pl/~pba/pdfdoc/Numerical_Optimization.pdf
        void MatVecHessian(const VectorXd& x, VectorXd* g);
        // The same as the previous method but for the inverse hessian estimation
        void MatVecInverseHessian(const VectorXd& x, VectorXd* g);
        subBFGSBoxBoundBase& operator = (const subBFGSBoxBoundBase& orig);
        subBFGSBoxBoundBase(const subBFGSBoxBoundBase& orig);
};
