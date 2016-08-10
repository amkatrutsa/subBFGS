#pragma once
#include "subBFGSBase.hpp"

class subBFGSTest1 : public subBFGSBase {
    public:
        subBFGSTest1(double eps, int k_max, double h);
        bool init();
        virtual ~subBFGSTest1();
    private:
        double ComputeObjective(const VectorXd& w);
        bool ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g);
        bool ComputeSubgradient(VectorXd* g);
        bool LineSearchStep();
        subBFGSTest1(const subBFGSTest1& orig);
        subBFGSTest1& operator = (const subBFGSTest1& orig);
};


