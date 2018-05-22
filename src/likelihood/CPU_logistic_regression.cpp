#include "CPU_logistic_regression.hpp"

VectorXd CPU_LogisticRegression::train(int n_iter,int mini_batch,double alpha,double step_size){
	VectorXd log_likelihood=VectorXd::Zero(n_iter);
	int num_batches=this->rows/mini_batch;
	VectorXd momemtum=VectorXd::Zero(this->dim);
	cout << "     Epoch     |     Loss        " << endl;
	for(int i=0;i<n_iter;i++){
		this->preCompute(i,mini_batch);
		log_likelihood(i)=-this->logPosterior();
		VectorXd gradient=this->computeGradient();
		if (i % num_batches == 0) cout << " " <<  i/num_batches << "         " << log_likelihood(i)  << "|  " << endl;
		momemtum*=alpha;
		momemtum-=(1.0-alpha)*gradient;
		this->weights+=momemtum*step_size;
		if(this->with_bias) this->bias-=step_size*this->grad_bias;
	}
	return log_likelihood;
}

void CPU_LogisticRegression::preCompute(int iter,int mini_batch){
	int num_batches=this->rows/mini_batch; 
	int idx = iter % num_batches;
	int start = idx * mini_batch;
	int end = (idx + 1) * mini_batch;
	this->X_slice=this->X_train->block(start,0,mini_batch,this->dim);
	this->y_slice=this->Y_train->segment(start,mini_batch);
	this->eta = (this->X_slice * this->weights);
	if(this->with_bias) this->eta.noalias()=(this->eta.array()+this->bias).matrix();
	this->phi = this->sigmoid(this->eta);
}

VectorXd CPU_LogisticRegression::computeGradient(){
	VectorXd E_d=this->phi-this->y_slice;
	VectorXd E_w=this->weights*this->lambda;
	//VectorXd grad=VectorXd::Zero(this->dim);
	//#pragma omp parallel for schedule(static)
	//for(int d=0;d<this->dim;d++){
	//	grad[d]=X_t.col(d).cwiseProduct(E_d).mean()+E_w[d];
	//}
	//VectorXd grad=(1.0/mini_batch)*X_t.transpose() * E_d + E_w;
	VectorXd grad=this->X_slice.transpose() * E_d + E_w;
	if(this->with_bias) this->grad_bias= (E_d.mean()+this->bias*this->lambda);
	return grad;
}

VectorXd CPU_LogisticRegression::predict(MatrixXd &_X_test,bool prob, bool data_processing){
	if (data_processing){
		if (this->normalization) tools.testNormalization(_X_test,this->featureMax,this->featureMin);
		if (this->standardization) tools.testStandardization(_X_test,this->featureMean,this->featureStd);
	}
	VectorXd eta_test = (_X_test)*this->weights;
	if(this->with_bias) eta_test.noalias()=(eta_test.array()+this->bias).matrix();
	VectorXd phi_test=this->sigmoid(eta_test);
	if(!prob){
		phi_test.noalias() = phi_test.unaryExpr([](double elem){
	    	return (elem > 0.5) ? 1.0 : 0.0;
		});		
	}
	return phi_test;
}
