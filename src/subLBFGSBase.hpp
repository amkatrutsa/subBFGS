// Copyright 2015 Alexandr Katrutsa, INRIA
#pragma once

#include <Eigen/Dense>
#include <vector>

typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;

// This class implements the subgradient BFGS (subBFGS) method for non-smooth function
// from the paper http://www.jmlr.org/papers/volume11/yu10a/yu10a.pdf
// To use this class, user has to create own class, public inherited from this class,
// and provide methods, which are pure virtual, and methods to get
// the objective or parameter vector if it is needed
class subLBFGSBase {
    public:
        subLBFGSBase(double epsilon, int k_max, double h);
        virtual ~subLBFGSBase();
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
//        MatrixXd B_;    // estimation of the inverse hessian
//        MatrixXd H_;    // estimation of the hessian
        double obj_;    // objective value
        int num_iter_;  // number of iteration after that converged
        const double EPS = 1e-20;  // tolerance of the non-zero subgradient
        double step_tol_;   // tolerance of the size step
        const int MAX_NUM_ITER_;
        const double OBJ_TOL_;
        // These vectors store the history of y, s and ro
        std::vector<VectorXd> history_y_;
        std::vector<VectorXd> history_s_;
        std::vector<double> history_ro_;
        // These vectors store some specific vectors to fast multiply hessian by vector
        // More details see in 
        // http://users.iems.northwestern.edu/~nocedal/PDFfiles/representations.pdf Section 4.2
        std::vector<VectorXd> history_a_;
        std::vector<VectorXd> history_b_;
        bool CheckWolfeConditions();
        bool CheckStep();
    private:
        virtual bool init() = 0;
        virtual bool LineSearchStep() = 0;
        virtual bool ComputeSubgradient(VectorXd* g) = 0;
        virtual double ComputeObjective(const VectorXd& w) = 0;
        virtual bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) = 0;
        virtual bool DescentDirection(const VectorXd& g);
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
        void UpdateBH(const VectorXd& s, const VectorXd& y);
        double CalculateEps(const std::vector<double>& pg_asup,
                            const std::vector<double>& pg_, double pg);
        bool CheckDescentDirection();
        bool LineSearch();
        // Clear all history vectors
        void Restart();
        // Print infinum and supremum of the objective in the found optimum vector
        virtual void PrintInfSupSubgrad();
        subLBFGSBase(const subLBFGSBase& orig);
        subLBFGSBase& operator = (const subLBFGSBase& origin);
};

