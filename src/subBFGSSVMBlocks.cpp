// Copyright 2015 Alexandr Katrutsa, INRIA

#include "subBFGSSVMBlocks.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>

subBFGSSVMBlocks::subBFGSSVMBlocks(double epsilon, int k_max, double h,
                                  const std::string& filename) :
                subBFGSBase(epsilon, k_max, h), block_list_filename_(filename),
                model_filename_("param_trainset8_C10.txt"), X_(NULL), y_(NULL),
                C_(NULL), cost_(10), total_vectors_(0) {}

subBFGSSVMBlocks::subBFGSSVMBlocks(double epsilon, int k_max, double h,
                                   const Parameters& p) :
                subBFGSBase(epsilon, k_max, h),
                block_list_filename_(p.blocksFile), model_filename_(p.modelFile),
                X_(NULL), y_(NULL), C_(NULL), cost_(p.C), total_vectors_(0) {}

bool subBFGSSVMBlocks::init() {
    assert(cost_ > 0);
    // Read the list of blocks filenames
    if (!ReadBlocksList()) {
        std::cout << "Can not read list of blocks filenames from " <<
                    block_list_filename_ << std::endl;
        return false;
    }
    // Allocate all necessary memory
    X_ = new MatrixXd[num_blocks_];
    y_ = new VectorXd[num_blocks_];
    C_ = new VectorXd[num_blocks_];
    X_natives_.resize(num_blocks_, N_);
    // Dimension of the problem is dimension of the vector parameter + number of blocks,
    // because every block has its own intercept b
    N_ += num_blocks_;
    // Read every block sequentially
    for (size_t block_idx = 0; block_idx < num_blocks_; ++block_idx) {
        if (!ReadBinaryBlock(block_idx)) {
            std::cout << "Can not read block " << block_idx <<
                    " from the file " << blocks_filenames_[block_idx] << std::endl;
            return false;
        }
    }
    // Initial value of the parameter vector
    w_ = VectorXd::Zero(N_);
    B_ = MatrixXd::Identity(N_, N_);
    return true;
}

bool subBFGSSVMBlocks::ReadBlocksList() {
    std::ifstream blocks_list;
    blocks_list.open(block_list_filename_.c_str(), std::ifstream::in);
    if (!blocks_list.is_open()) {
        std::cout << "Can not open file " << block_list_filename_ << std::endl;
        return false;
    }
    blocks_list >> num_blocks_;
    blocks_list >> N_;
    vector_dimension_ = N_;
    blocks_filenames_.reserve(num_blocks_);
    std::string block_filename;
    for (blocks_list >> block_filename; !blocks_list.eof(); blocks_list >> block_filename)
        blocks_filenames_.push_back(block_filename);
    blocks_list.close();
    if (num_blocks_ != blocks_filenames_.size()) {
        std::cout << "Number of blocks given in the file (" << num_blocks_ <<
                     ") does not correspond to the read number of blocks (" <<
                     blocks_filenames_.size() << ")" << std::endl;
        blocks_list.close();
        return false;
    }
    blocks_list.close();
    return true;
}

bool subBFGSSVMBlocks::ReadBinaryBlock(int block_idx) {
    std::string current_block_filename = blocks_filenames_[block_idx];
    FILE* f = fopen(current_block_filename.c_str(), "rb");
    if (!f) {
        std::cout << "Can not open file " << current_block_filename << std::endl;
        return false;
    }
    if (block_idx == 0) {
        printf("The first block is %s\n", current_block_filename.c_str());
    }
    int num_vectors = 0;
    int dim = 0;
    size_t flag = 0;
    // Read how many vectors one must read from this file
    flag = fread(&(num_vectors), sizeof(int), 1, f);
    if (flag != 1) {
        std::cout << "Error in reading number of vectors in the file " <<
                    current_block_filename << std::endl;
        fclose(f);
        return false;
    }
    // Increment the number of vectors
    total_vectors_ += num_vectors;
    // Read dimension of vectors
    flag = fread(&(dim), sizeof(int), 1, f);
    if (flag != 1) {
        std::cout << "Error in reading dimension of vectors in the file " <<
                    current_block_filename << std::endl;
        fclose(f);
        return false;
    }
    if (dim != vector_dimension_) {
        std::cout << "Dimension in the block " << current_block_filename <<
                    " is not equal to the given dimension: " <<
                    dim << " != " << vector_dimension_ << std::endl;
        fclose(f);
        return false;
    }
    // Resize all storages for current block
    X_[block_idx].resize(num_vectors, dim);
    y_[block_idx].resize(num_vectors);
    C_[block_idx].resize(num_vectors);
    char tmp;
    int class_label = 0;
    int num_positive_vectors = 0;
    int num_negative_vectors = 0;
    // Read vectors and class labels from the current block
    for (int i = 0; i < num_vectors; ++i) {
        fread(&(tmp), sizeof(char), 1, f);  // read separator
        fread(&(class_label), sizeof(int), 1, f);   // read class label

        if (class_label == 1) {
            y_[block_idx][i] = 1;
            ++num_positive_vectors;
        } else if (class_label == -1) {
            y_[block_idx][i] = -1;
            ++num_negative_vectors;
        } else {
            std::cout << "Unknown class label of the vector " << i <<
                         " in the block " << current_block_filename << std::endl;
            fclose(f);
            return false;
        }
        for (int k = 0; k < dim; ++k)
            fread(&(X_[block_idx](i, k)), sizeof(double), 1, f);    // read element
    }
    // Set the scaled cost C for every vector in this block
    if (num_vectors != num_positive_vectors + num_negative_vectors) {
        std::cout << "Sum of negative and positive vectors is not equal "
                     "the total number of vectors in block " << current_block_filename << std::endl;
        fclose(f);
        return false;
    }
    double Cpos = static_cast<double>(num_positive_vectors) / static_cast<double>(num_vectors);
    double Cneg = static_cast<double>(num_negative_vectors) / static_cast<double>(num_vectors);
    for (int i = 0; i < num_vectors; ++i) {
        if (y_[block_idx][i] == 1)
            C_[block_idx][i] = cost_ * Cpos;
        else
            C_[block_idx][i] = cost_ * Cneg;
    }
    if (block_idx == 0) {
        C_[block_idx] *= 1000;
    }
    X_natives_.row(block_idx) = X_[block_idx].row(0);
    VectorXd native_vector = X_[block_idx].row(0);
    for (int i = 0; i < num_vectors; ++i) {
        X_[block_idx].row(i) = X_[block_idx].row(i) - native_vector.transpose();
    }
    fclose(f);
    return true;
}

bool subBFGSSVMBlocks::LineSearchStep() {
    if (!LineSearchIncremental()) {
        std::cout << "Error in computing step!" << std::endl;
        return false;
    }
    return true;
}

bool subBFGSSVMBlocks::LineSearchIncremental() {
    double norm_p2 = p_.tail(vector_dimension_).dot(p_.tail(vector_dimension_));
    double wp =  w_.tail(vector_dimension_).dot(p_.tail(vector_dimension_));
    std::vector<double> subdif_eta;
    subdif_eta.reserve(total_vectors_);
    std::vector<sorted_blocks_idx> subdif_eta_idx;
    subdif_eta_idx.reserve(total_vectors_);
    VectorXd f_dot_product;
    VectorXd f_vector_score;
    VectorXd* f = new VectorXd[num_blocks_];
    VectorXd df_dot_product;
    VectorXd df_vector_score;
    VectorXd* delta_f = new VectorXd[num_blocks_];
    for (size_t i = 0; i < num_blocks_; ++i) {
        f_dot_product = X_[i] * w_.tail(vector_dimension_);
        f_vector_score = f_dot_product.array() + w_[i];
        df_dot_product = X_[i] * p_.tail(vector_dimension_);
        df_vector_score = df_dot_product.array() + p_[i];
        f[i].resize(f_vector_score.rows());
        delta_f[i].resize(df_vector_score.rows());
        f[i] = y_[i].cwiseProduct(f_vector_score);
        delta_f[i] = y_[i].cwiseProduct(df_vector_score);
        for (int j = 0; j < f[i].rows(); ++j) {
            if (delta_f[i][j] != 0) {
                subdif_eta.push_back((1 - f[i][j]) / delta_f[i][j]);
                sorted_blocks_idx cur_eta_idx;
                cur_eta_idx.total_eta_idx = 0;
                cur_eta_idx.block_idx = i;
                cur_eta_idx.vector_idx = j;
                subdif_eta_idx.push_back(cur_eta_idx);
            }
        }
    }
    std::vector<double> eta;
    eta.reserve(subdif_eta.size());
    std::vector<sorted_blocks_idx> sorted_idx;
    int positive_eta_counter = 0;
    for (size_t i = 0; i < subdif_eta.size(); ++i) {
        if (subdif_eta[i] > 0) {
            eta.push_back(subdif_eta[i]);
            subdif_eta_idx[i].total_eta_idx = positive_eta_counter;
            sorted_idx.push_back(subdif_eta_idx[i]);
            ++positive_eta_counter;
        }
    }
    ArgSort(eta, &sorted_idx);
    double init_grad_norm_w = wp + eta[sorted_idx[0].total_eta_idx] * norm_p2;
    double init_grad_misclass = 0;
    for (size_t i = 0; i < num_blocks_; ++i) {
        for (int j = 0; j < f[i].rows(); ++j) {
            if (f[i][j] + eta[sorted_idx[0].total_eta_idx] * delta_f[i][j] <= 1) {
                init_grad_misclass -= C_[i][j] * delta_f[i][j];
            }
        }
    }
    double init_grad = init_grad_norm_w + init_grad_misclass;
    double init_grad_sup = 0;
    double init_grad_inf = 0;
    if (delta_f[sorted_idx[0].block_idx][sorted_idx[0].vector_idx] < 0) {
        init_grad_sup = init_grad - C_[sorted_idx[0].block_idx][sorted_idx[0].vector_idx] *
                        delta_f[sorted_idx[0].block_idx][sorted_idx[0].vector_idx];
        init_grad_inf = init_grad;
    } else {
        init_grad_sup = init_grad + C_[sorted_idx[0].block_idx][sorted_idx[0].vector_idx] *
                        delta_f[sorted_idx[0].block_idx][sorted_idx[0].vector_idx];
        init_grad_inf = init_grad;
    }
//    printf("Left grad in %e = %e\n", eta[sorted_idx[0].total_eta_idx], init_grad_inf);
//    printf("Right grad in %e = %e\n", eta[sorted_idx[0].total_eta_idx], init_grad_sup);
    // Check supremum and infimum of the subgradient in the first subdifferential point
    // If infimum and supremum absolute values are small enough then current
    // vector w_ is already optimal
    if ((fabs(init_grad_inf) < EPS) && (fabs(init_grad_sup) < EPS)) {
        std::cout << "The module of infinum and supremum are less than " << EPS << std::endl;
        eta_ = 0;
        delete[] f;
        delete[] delta_f;
        return true;
    }
    // If 0 lies between infimum and supremum, then this subdifferential point is
    // an optimum step
    if ((init_grad_inf < 0) && (init_grad_sup > 0)) {
        std::cout << "The first subdifferential point is a step" << std::endl;
        eta_ = eta[sorted_idx[0].total_eta_idx];
        return true;
    }
    // If both infimum and supremum are positive, then the optimum step is
    // between 0 and minimum subdifferential point or in the border of this segment
    if ((init_grad_inf > 0) && (init_grad_sup > 0)) {
        double test_eta = eta[sorted_idx[0].total_eta_idx] * 0.5;
        eta_ = -wp;
        for (size_t i = 0; i < num_blocks_; ++i) {
            for (int j = 0; j < f[i].rows(); ++j) {
                if (f[i][j] + test_eta * delta_f[i][j] < 1)
                    eta_ += C_[i][j] * delta_f[i][j];
            }
        }
        eta_ /= norm_p2;
        double left_obj = ComputeObjective(w_);
        double right_obj = ComputeObjective(w_ + eta[sorted_idx[0].total_eta_idx] * p_);
        // If eta_ lies between 0 and eta[sorted_idx[0]], check where is
        // the objective smaller: in 0, in eta_ or in eta[sorted_idx[0]]
        if ((eta_ >= 0) && (eta_ <= eta[sorted_idx[0].total_eta_idx])) {
            double middle_obj = ComputeObjective(w_ + eta_ * p_);
            if ((left_obj < right_obj) && (left_obj < middle_obj)) {
                std::cout << "Objective in 0 is less than in the first subdifferential point and average point" << std::endl;
                eta_ = 0;
                delete[] f;
                delete[] delta_f;
                return true;
            }
            if ((middle_obj < left_obj) && (middle_obj < right_obj)) {
                std::cout << "Step size lies between 0 and first subdiffirential point" << std::endl;
                delete[] f;
                delete[] delta_f;
                return true;
            }
            if ((right_obj < left_obj) && (right_obj < middle_obj)) {
                eta_ = eta[sorted_idx[0].total_eta_idx];
                std::cout << "Step size is the first subdiffirential point" << std::endl;
                delete[] f;
                delete[] delta_f;
                return true;
            }
        } else {
            if (right_obj < left_obj) {
                std::cout << "The step is the first subdifferential point" << std::endl;
                eta_ = eta[sorted_idx[0].total_eta_idx];
            } else {
                std::cout << "Objective in 0 is less than in the first subdifferential point" << std::endl;
                
                eta_ = 0;
            }
            delete[] f;
            delete[] delta_f;
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
        cur_grad_norm_w += (eta[sorted_idx[i].total_eta_idx] - eta[sorted_idx[i-1].total_eta_idx]) * norm_p2;
        double cur_grad = cur_grad_norm_w + cur_grad_misclass;
        if (delta_f[sorted_idx[i].block_idx][sorted_idx[i].vector_idx] < 0) {
            cur_grad_sup = cur_grad - C_[sorted_idx[i].block_idx][sorted_idx[i].vector_idx] *
                            delta_f[sorted_idx[i].block_idx][sorted_idx[i].vector_idx];
            cur_grad_inf = cur_grad;
        } else {
            cur_grad_sup = cur_grad + C_[sorted_idx[i].block_idx][sorted_idx[i].vector_idx] *
                            delta_f[sorted_idx[i].block_idx][sorted_idx[i].vector_idx];
            cur_grad_inf = cur_grad;
        }
//        std::cout << "Left grad in " << eta[sorted_idx[i].total_eta_idx] << " = " << cur_grad_inf << std::endl;
//        std::cout << "Right grad in " << eta[sorted_idx[i].total_eta_idx] << " = " << cur_grad_sup << std::endl;
        // Analyse the supremum and infimum of the current subdifferentiabe point
        if ((cur_grad_inf < 0) && (cur_grad_sup > 0)) {
            eta_ = eta[sorted_idx[i].total_eta_idx];
            std::cout << "The step is the " << i << "-th subdifferential point" << std::endl;
            delete[] f;
            delete[] delta_f;
            return true;
        }
        if ((init_grad_sup < 0) && (cur_grad_inf > 0)) {
            double test_eta = 0.5 * (eta[sorted_idx[i-1].total_eta_idx] + eta[sorted_idx[i].total_eta_idx]);
            eta_ = -wp;
            for (size_t i = 0; i < num_blocks_; ++i) {
                for (int j = 0; j < f[i].rows(); ++j) {
                    if (f[i][j] + test_eta * delta_f[i][j] < 1)
                        eta_ += C_[i][j] * delta_f[i][j];
                }
            }
            eta_ /= norm_p2;
            double left_obj = ComputeObjective(w_ + eta[sorted_idx[i-1].total_eta_idx] * p_);
            double right_obj = ComputeObjective(w_ + eta[sorted_idx[i].total_eta_idx] * p_);
            if ((eta_ <= eta[sorted_idx[i].total_eta_idx]) && (eta_ >= eta[sorted_idx[i-1].total_eta_idx])) {
                double middle_obj = ComputeObjective(w_ + eta_ * p_);
                if ((left_obj < right_obj) && (left_obj < middle_obj)) {
                    eta_ = eta[sorted_idx[i-1].total_eta_idx];
                    std::cout << "The step is the (" << i-1 << ")-th subdifferential point" << std::endl;
                    delete[] f;
                    delete[] delta_f;
                    return true;
                }
                if ((middle_obj < left_obj) && (middle_obj < right_obj)) {
                    std::cout << "The step lies between " << (i-1) << "-th and " << i <<
                                "-th subdifferential points" << std::endl;
                    delete[] f;
                    delete[] delta_f;
                    return true;
                }
                if ((right_obj < left_obj) && (right_obj < middle_obj)) {
                    eta_ = eta[sorted_idx[i].total_eta_idx];
                    std::cout << "The step is the " << i << "-th subdifferential point" << std::endl;
                    delete[] f;
                    delete[] delta_f;
                    return true;
                }
            } else {
                if (left_obj < right_obj) {
                    eta_ = eta[sorted_idx[i-1].total_eta_idx];
                    std::cout << "The step is the (" << i-1 << ")-th subdifferential point" << std::endl;
                    delete[] f;
                    delete[] delta_f;
                    return true;
                } else {
                    eta_ = eta[sorted_idx[i].total_eta_idx];
                    std::cout << "The step is the " << i << "-th subdifferential point" << std::endl;
                    delete[] f;
                    delete[] delta_f;
                    return true;
                }
            }
        }
//        init_grad_inf = cur_grad_inf;
        init_grad_sup = cur_grad_sup;
        cur_grad_misclass = cur_grad_sup - cur_grad_norm_w;
    }
    std::cout << "Can not find optimal step!" << std::endl;
    delete[] f;
    delete[] delta_f;
    return false;
}

bool subBFGSSVMBlocks::ComputeSubgradient(VectorXd* g) {
    assert(g->rows() == N_);
    g->tail(vector_dimension_) = w_.tail(vector_dimension_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        assert(X_[i].cols() == vector_dimension_);
        VectorXd dot_products = X_[i] * w_.tail(vector_dimension_);
        VectorXd score_vector = dot_products.array() + w_[i];
        VectorXd loss = 1 - y_[i].cwiseProduct(score_vector).array();
        for (int j = 0; j < loss.rows(); ++j) {
            if (loss[j] > 0) {
                g->tail(vector_dimension_) -= C_[i][j] * y_[i][j] * X_[i].row(j).transpose();
                (*g)[i] -= C_[i][j] * y_[i][j];
            }
        }
    }
    return true;
}

void subBFGSSVMBlocks::PrintInfSupSubgrad() {
    VectorXd inf_g = VectorXd::Zero(N_);
    VectorXd sup_g = VectorXd::Zero(N_);
    inf_g.tail(vector_dimension_) = w_.tail(vector_dimension_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        assert(X_[i].cols() == vector_dimension_);
        VectorXd dot_products = X_[i] * w_.tail(vector_dimension_);
        VectorXd score_vector = dot_products.array() + w_[i];
        VectorXd loss = 1 - y_[i].cwiseProduct(score_vector).array();
        for (int j = 0; j < loss.rows(); ++j) {
            if (loss[j] > 0) {
                inf_g.tail(vector_dimension_) -= C_[i][j] * y_[i][j] * X_[i].row(j).transpose();
                inf_g[i] -= C_[i][j] * y_[i][j];
            }
            sup_g = inf_g;
            if (loss[j] == 0) {
                if (C_[i][j] * y_[i][j] > 0) {
                    inf_g[i] -= C_[i][j] * y_[i][j];
                }
                else {
                    sup_g[i] -= C_[i][j] * y_[i][j];
                }
                inf_g.tail(vector_dimension_) -= C_[i][j] * y_[i][j] * X_[i].row(j).transpose();
            }
        }
    }
    printf("Infimum Supremum:\n");
    for (int i = 0; i < inf_g.rows(); ++i)
        printf("%f %f\n", inf_g[i], sup_g[i]);
}

double subBFGSSVMBlocks::ComputeObjective(const VectorXd& w) {
    obj_ = 0.5 * w.tail(vector_dimension_).dot(w.tail(vector_dimension_));
    for (size_t i = 0; i < num_blocks_; ++i) {
        assert(X_[i].cols() == vector_dimension_);
        VectorXd dot_products = X_[i] * w.tail(vector_dimension_);
        VectorXd score_vector = dot_products.array() + w[i];
        VectorXd loss = 1 - y_[i].cwiseProduct(score_vector).array();
        for (int j = 0; j < loss.rows(); ++j) {
            if (loss[j] > 0)
                obj_ += C_[i][j] * loss[j];
        }
    }
    return obj_;
}

bool subBFGSSVMBlocks::ArgSup(const VectorXd& p, const VectorXd& w, VectorXd* g) {
    assert(g->rows() == N_);
    assert(p.rows() == N_);
    g->head(num_blocks_) = VectorXd::Zero(num_blocks_);
    g->tail(vector_dimension_) = w_.tail(vector_dimension_);
    for (size_t i = 0; i < num_blocks_; ++i) {
        assert(X_[i].cols() == vector_dimension_);
        VectorXd dot_products = X_[i] * w.tail(vector_dimension_);
        VectorXd score_vector = dot_products.array() + w[i];
        VectorXd loss = 1 - y_[i].cwiseProduct(score_vector).array();
        for (int j = 0; j < loss.rows(); ++j) {
            if (loss[j] > 0) {
                g->tail(vector_dimension_) -= C_[i][j] * y_[i][j] * X_[i].row(j).transpose();
                (*g)[i] -= C_[i][j] * y_[i][j];
            } else if (loss[j] == 0) {
                if (y_[i][j] * X_[i].row(j).dot(p.tail(vector_dimension_)) < 0)
                    g->tail(vector_dimension_) -= C_[i][j] * y_[i][j] * X_[i].row(j).transpose();
                if (p[i] * y_[i][j] < 0)
                    (*g)[i] -= C_[i][j] * y_[i][j];
            }
        }
    }
    return true;
}

void subBFGSSVMBlocks::ArgSort(const std::vector<double>& eta, std::vector<sorted_blocks_idx>* sorted_idx) {
    assert(sorted_idx->size() == eta.size());
    std::sort(sorted_idx->begin(), sorted_idx->end(),
             [&eta] (sorted_blocks_idx i1, sorted_blocks_idx i2) {
                    return eta[i1.total_eta_idx] < eta[i2.total_eta_idx]; });
}

double subBFGSSVMBlocks::get_objective() {
    double norm_w2 = w_.tail(vector_dimension_).dot(w_.tail(vector_dimension_));
    printf("||w||^2 / 2 = %.6f\n", 0.5 * norm_w2);
    double misclass_error = 0;
    for (size_t i = 0; i < num_blocks_; ++i) {
        VectorXd dot_product = X_[i] * w_.tail(vector_dimension_);
        VectorXd score_vector = dot_product.array() + w_[i];
        VectorXd loss = 1 - y_[i].cwiseProduct(score_vector).array();
        for (int j = 0; j < loss.rows(); ++j) {
            if (loss[j] > 0)
                misclass_error += C_[i][j] * loss[j];
        }
    }
    printf("Misclassification error = %.6f\n", misclass_error);
    return 0.5 * norm_w2 + misclass_error;
}

bool subBFGSSVMBlocks::write_to_file() {
    std::ofstream output_file(model_filename_);
    if (!output_file.is_open()) {
        std::cout << "Can not open file " << model_filename_ << std::endl;
        return false;
    }
    output_file << num_blocks_ << " " << w_.rows() << " " << vector_dimension_ << "\n";
    for (int i = 0; i < w_.rows(); ++i) {
        output_file << w_[i] << "\n";
    }
    output_file.close();
    return true;
}

//void subBFGSSVMBlocks::ProjectParamVector() {}
//void subBFGSSVMBlocks::ProjectParamVector() {
//    for (int i = w_.rows() - NUM_ENTROPY_TERMS_; i < w_.rows(); ++i) {
//        if (w_[i] < 0)
//            w_[i] = 0;
//    }
//}

double subBFGSSVMBlocks::compute_accuracy() {
    double current_score = 0;
    double num_correct = 0;
    int idx_min_score = 0;
    for (size_t i = 0; i < num_blocks_; ++i) {
        double min_score = std::numeric_limits<double>::infinity();
        for (int j = 0; j < y_[i].rows(); ++j) {
            current_score = X_[i].row(j).dot(w_.tail(vector_dimension_));
            if (current_score < min_score) {
                min_score = current_score;
                idx_min_score = j;
            }
        }
        if (y_[i][idx_min_score] == -1)
            ++num_correct;
    }
    return num_correct / num_blocks_;
}

void subBFGSSVMBlocks::PrintInformationCurrentIter() {
    double current_accuracy = compute_accuracy();
    printf("Current accuracy = %e\n", current_accuracy);
    int num_neg_entropy_term = 0;
    VectorXd entropy_terms = w_.tail(NUM_ENTROPY_TERMS_);
    for (size_t i = 0; i < entropy_terms.rows(); ++i) {
        if (entropy_terms[i] < 0)
            ++num_neg_entropy_term;
    }
    printf("Number of negative entropy terms = %d\n", num_neg_entropy_term);
}

bool subBFGSSVMBlocks::check_solution(const std::string& solution_filename) {
    std::ifstream solution_file(solution_filename);
    if (!solution_file.is_open()) {
        std::cout << "Can not open file " << solution_filename << std::endl;
        return false;
    }
    int temp, dim, vec_dim;
    solution_file >> temp >> dim >> vec_dim;
    VectorXd w(dim);
    for (int i = 0; i < dim; ++i)
        solution_file >> w[i];
    double current_score = 0;
    double num_correct = 0;
    int idx_min_score = 0;
    for (size_t i = 0; i < num_blocks_; ++i) {
        double min_score = std::numeric_limits<double>::infinity();
        for (int j = 0; j < y_[i].rows(); ++j) {
            current_score = (X_[i].row(j) + X_natives_.row(i)).dot(w.tail(vec_dim));
//            current_score = X_[i].row(j).dot(w.tail(vec_dim));
//            std::cout << "Score of block " << i << " vector " << j << " = " << current_score << std::endl;
            if (current_score < min_score) {
                min_score = current_score;
                idx_min_score = j;
            }
        }
        if (y_[i][idx_min_score] == -1)
            ++num_correct;
    }
    printf("Accuracy of the solution from the file %s = %.6f\n", solution_filename.c_str(), num_correct / num_blocks_);
    return true;
}

subBFGSSVMBlocks::~subBFGSSVMBlocks() {
    if (X_)
        delete[] X_;
    if (y_)
        delete[] y_;
    if (C_)
        delete[] C_;
}

