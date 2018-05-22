//Author: Diego Vergara
#include "softmax_regression.hpp"

SoftmaxRegression::SoftmaxRegression(){
	this->normalization= false;
	this->standardization=false;
	this->with_bias = false;
}

void SoftmaxRegression::init(bool _normalization, bool _standardization,bool _with_bias){
	std::setprecision(10);
	this->normalization=_normalization;
	this->standardization=_standardization;
	this->with_bias = _with_bias;
}

void SoftmaxRegression::init(MatrixXd &_X,VectorXd &_Y,double _reg, bool _normalization,bool _standardization,bool _with_bias){
	this->initialized = true;
	this->with_bias = _with_bias;
	srand((unsigned int) time(0));
	this->normalization=_normalization;
	this->standardization=_standardization;
	this->reg=_reg;
 	this->X_train = &_X;
 	this->Y_train = &_Y;
 	tools.dataPermutation(*this->X_train,*this->Y_train);
 	if(this->normalization) tools.dataNormalization(*this->X_train,this->featureMax,this->featureMin);
 	if(this->standardization) tools.dataStandardization(*this->X_train,this->featureMean,this->featureStd);
 	this->rows = this->X_train->rows();
	this->dim = this->X_train->cols();
	if (this->with_bias){ //basis map: identity
		this->dim++;
		this->X_train->conservativeResize(NoChange,this->dim);
		this->X_train->col(this->dim-1) = VectorXd::Constant(this->rows, 1.0/this->dim);
	}
	this->classes.resize(0);
 	this->labelBinarize = tools.binarizeLabels(*this->Y_train, this->classes);
 	this->n_classes = classes.size();
	this->weights = MatrixXd::Random(this->n_classes, this->dim)/this->dim;
	this->eta = MatrixXd::Zero(this->rows, n_classes);
	this->phi = MatrixXd::Zero(this->rows, n_classes);
 }

int SoftmaxRegression::getNClasses(){
	return this->n_classes;
}

VectorXd SoftmaxRegression::getClasses(){
	return this->classes;
}

void SoftmaxRegression::preCompute(){
	this->eta = (*this->X_train * this->weights.transpose());
}

MatrixXd SoftmaxRegression::logsoftmax(MatrixXd &eta){
	VectorXd m = eta.rowwise().maxCoeff();
	VectorXd log_sum_exp = ((((eta.colwise() - m).array().exp()).rowwise().sum()).log()).array() + m.array();  // logsumexp
	return ((((eta.colwise() - m).array().exp()).log()).matrix().colwise()  - log_sum_exp);
}

MatrixXd SoftmaxRegression::softmax(MatrixXd &eta){
	VectorXd exp_sum = eta.array().exp().rowwise().sum(); 
	return eta.array().exp().colwise() / exp_sum.array();

}

double SoftmaxRegression::logLikelihood(){
	//this->phi = this->logsoftmax(this->eta);
	this->phi = this->softmax(this->eta);
	//double log_likelihood =  -(this->labelBinarize.array()*this->phi.array()).mean() - ((this->weights * this->weights.transpose()).trace() * (this->reg/(double)this->weights.rows()));
	double log_likelihood =  -(this->labelBinarize.array()*this->phi.array().log()).mean() - ((this->weights * this->weights.transpose()).trace() * (this->reg/(double)this->weights.rows()));
	return log_likelihood;
}

/*double SoftmaxRegression::logPrior(){
    double prior = -log(sqrt(2*M_PI))-this->weights.squaredNorm()/(2*this->reg);
    //if(this->with_bias) prior -=  pow(this->bias,2)/(2*this->reg);
	return prior;
}*/

double SoftmaxRegression::logPosterior(){
	//VectorXd log_likelihood=logLikelihood();
	return logLikelihood();
	//double log_prior = logPrior();
    //return log_likelihood.array() + log_prior;
}

void SoftmaxRegression::setWeights(MatrixXd &_W){
	this->weights=_W;
}

MatrixXd SoftmaxRegression::getWeights(){
	return this->weights;
}

void SoftmaxRegression::setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing){
	this->X_train = &_X;
 	this->Y_train = &_Y;
	tools.dataPermutation(*this->X_train,*this->Y_train);
 	this->rows = this->X_train->rows();
	this->dim = this->X_train->cols();
	if(_preprocesing){
		if(this->normalization) tools.dataNormalization(*this->X_train,this->featureMax,this->featureMin);
 		if(this->standardization) tools.dataStandardization(*this->X_train,this->featureMean,this->featureStd);
	}
	if (this->with_bias){ //basis map: identity
		this->dim++;
		this->X_train->conservativeResize(NoChange,this->dim);
		this->X_train->col(this->dim-1) = VectorXd::Constant(this->rows, 1.0/this->dim);
	}
 	this->labelBinarize = tools.binarizeLabels(_Y, this->classes);
 	this->n_classes = classes.size();
	
}
