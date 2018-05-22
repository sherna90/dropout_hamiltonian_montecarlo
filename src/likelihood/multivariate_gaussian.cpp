#include "multivariate_gaussian.hpp"
MVNGaussian::MVNGaussian(){
    
}

MVNGaussian::MVNGaussian(VectorXd _mean, MatrixXd _cov){
    mean = _mean;
    cov = _cov;
}

MVNGaussian::MVNGaussian(MatrixXd &data, bool  _diag){
    /* Getting mean for every column */
    mean = data.colwise().mean();
    /* Covariance Matrix */
    if (_diag){
        VectorXd var = (data.rowwise() - mean.transpose()).array().square().colwise().sum() / (data.rows());
        if(var.all() >0 ){
            cov = var.asDiagonal();
            VectorXd temp = 1.0/var.array();
            invcov = (temp).asDiagonal();
        } 
        else{
            cout << "Warning: Zeros in Variance vector, set Diagonal Unitary Covariance" << endl;
            cov=VectorXd::Ones(data.cols()).asDiagonal();
            invcov = cov;      
        } 
        
    }
    else{
        MatrixXd centered = data.rowwise() - mean.transpose();
        cov = (centered.adjoint() * centered) / double(data.rows() - 1);
        invcov = cov.inverse();
    }

}

void MVNGaussian::generate(bool _cholesky){
    normX_solver = EigenMultivariateNormal<double>(mean,cov, _cholesky);
}

VectorXd MVNGaussian::getMean(void){
    return mean;
}

MatrixXd MVNGaussian::getCov(void){
    return cov;
}

MatrixXd MVNGaussian::getInvCov(void){
    return invcov;
}

void MVNGaussian::setMean(VectorXd &_mean){
    mean = _mean;
}

void MVNGaussian::setCov(MatrixXd &_cov){
    cov = _cov;
}

VectorXd MVNGaussian::sample(){
    return normX_solver.samples(1);
}

MatrixXd MVNGaussian::sample(int n_samples){
    return normX_solver.samples(n_samples);
}

VectorXd MVNGaussian::log_likelihood(MatrixXd data){
    double rows = data.rows();
    double cols = data.cols();
    VectorXd loglike = VectorXd::Zero(rows);
    /* Getting inverse matrix for 'cov' with Cholesky */
    LLT<MatrixXd> chol(cov);
    MatrixXd L = chol.matrixL();
    MatrixXd cov_inverse = L.adjoint().inverse() * L.inverse();
    double logdet=log(cov.determinant());
     for(unsigned i=0;i<rows;i++){
        VectorXd tmp1 = data.row(i);
        tmp1 -= mean;
        MatrixXd tmp2 = tmp1.transpose() * cov_inverse;
        tmp2 = tmp2 * tmp1;
        loglike(i) = -0.5 * tmp2(0,0) - (cols/2) * log(2*M_PI) -(0.5) * logdet;
    }
    //cout << loglike << endl;
    return loglike;
}
