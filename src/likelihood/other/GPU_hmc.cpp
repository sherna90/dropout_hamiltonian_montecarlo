//Author: Diego Vergara
#include "GPU_hmc.hpp"

void GPU_Hamiltonian_MC::init(MatrixXd &_X, VectorXd &_Y, double _lambda, int _warmup_iterations, int _iterations, double _step_size, int _num_step, bool _normalization, bool _standarization, double _path_length){
	this->lambda=_lambda;
	this->step_size = _step_size;
	this->num_step = _num_step;
	this->path_length = _path_length;
	if (this->path_length > 0.0) this->num_step = int(this->path_length/this->step_size);
	this->warmup_iterations = _warmup_iterations;
	this->X_train = &_X;
 	this->Y_train = &_Y;
	this->dim = _X.cols()+1; // + bias
	this->normalization = _normalization;
	this->standarization = _standarization;
    this->logistic_regression.init(_X, _Y, this->lambda, this->normalization, this->standarization, true);
    this->init_hmc = true;
    this->initialized = true;
    this->sampled = 0.0;
    this->accepted = 0.0;
    VectorXd mu = VectorXd::Zero(dim);
	MatrixXd cov = VectorXd::Ones(dim).asDiagonal();
	this->inv_cov = cov.inverse();
    this->multivariate_gaussian = MVNGaussian(mu, cov);
    this->multivariate_gaussian.generate();
    this->current_x = VectorXd::Random(this->dim);
    if (this->warmup_iterations >= 20) this->warmup();
    this->iterations = _iterations;

}

VectorXd GPU_Hamiltonian_MC::gradient(VectorXd &W){
	VectorXd grad(W.rows());
	if (this->init_hmc)
	{	
		VectorXd temp = W.tail(W.rows()-1);
		this->logistic_regression.setWeights(temp);
		this->logistic_regression.setBias(W(0));
		this->logistic_regression.preCompute();
		VectorXd gradWeights = this->logistic_regression.computeGradient();
		double gradBias = this->logistic_regression.getGradientBias();
		grad << gradBias, gradWeights;
		return grad;
	}
	else{
		return grad;
	}
}

double GPU_Hamiltonian_MC::logPosterior(VectorXd &W, bool precompute){
	double logPost = 0.0;
	if (this->init_hmc){
		VectorXd temp = W.tail(W.rows()-1);
		this->logistic_regression.setWeights(temp);
		this->logistic_regression.setBias(W(0));
		if(precompute) this->logistic_regression.preCompute();
		logPost = -this->logistic_regression.logPosterior();
		return logPost;
	}
	else{
		return logPost;
	}
}


void GPU_Hamiltonian_MC::run(bool warmup_flag, bool for_predict){
	if (!warmup_flag and !for_predict) cout << "Run" << endl;
	if (this->init_hmc){	

		VectorXd x = this->current_x;
		//bool accepted_flag = false;
		this->weights.resize(this->iterations, this->dim);
		
		//Hamiltonian
		double Hold;
		double Hnew;
		double Enew;
		this->logistic_regression.setGPUFlags(true, false); //GPU
		double Eold = this->logPosterior(x);

		VectorXd p;

		int n = 0;
		while (n < this->iterations){
			if(!for_predict) tools.printProgBar(n, this->iterations);

			p = initial_momentum();

			VectorXd xold = x;
			VectorXd pold = p;

			double epsilon = this->unif(this->step_size);

			if (this->path_length > 0.0) this->num_step = int(this->path_length/epsilon);

			this->logistic_regression.setGPUFlags(false, false); //GPU

			p.noalias() = p - 0.5*epsilon*this->gradient(x);

			//Leap Frogs
			for (int i = 0; i < this->num_step; ++i){
				x.noalias() = x + epsilon*p;
				if(i == (this->num_step-1)) p.noalias() = p - epsilon*this->gradient(x);
			}

			p.noalias() = p - 0.5*epsilon*this->gradient(x);

			if((n==(this->iterations-1))) this->logistic_regression.setGPUFlags(false, true); //GPU

			//Hamiltonian
			Enew = this->logPosterior(x, false);

			if (warmup_flag){
				Hnew = Enew + 0.5 * p.adjoint()*p;
				Hold = Eold + 0.5 * pold.adjoint()*pold;	
			}
			else{
				Hnew = Enew + 0.5 * (p.transpose()*this->inv_cov)*p;
				Hold = Eold + 0.5 * (pold.transpose()*this->inv_cov)*pold;
			}

			//Metropolis Hasting Correction
			double a = min(0.0, Hold - Hnew);
			if (log(random_uniform()) < a ){
				Eold = Enew;
				this->accepted++;
			}
			else{
				x = xold;	
			}
			
			this->weights.row(n) = x;

			this->sampled++;

			n = n+1;

		}
		this->logistic_regression.setGPUFlags(true, true); //GPU
		this->current_x = x;

		if(!for_predict)cout << endl;
		if(!for_predict)this->acceptace_rate();

	}
	else{
		cout << "Error: No initialized function"<< endl;
	}
}

VectorXd GPU_Hamiltonian_MC::predict(MatrixXd &X_test, bool prob, int samples, bool simulation){

	VectorXd predict;
	if (this->init_hmc){
		int partition = (int)this->iterations*0.5; 
		MatrixXd temp_weights = this->weights.block(partition,0 ,this->weights.rows()-partition, this->dim);
		this->mean_weights = temp_weights.colwise().mean();
		
		this->sampled = 0.0;
	    this->accepted = 0.0;
	    bool data_processing = true;

	    if(samples > 1){
			MatrixXd temp_predict(samples, X_test.rows());
			if (simulation){

				this->multivariate_gaussian = MVNGaussian(temp_weights);
				this->multivariate_gaussian.generate();

				/*this->multivariate_gaussian = MVNGaussian(temp_weights);
				VectorXd mu = VectorXd::Zero(dim);
				this->multivariate_gaussian.setMean(mu);
				this->multivariate_gaussian.generate();*/

				this->iterations = samples;
				this->run(false, true);
			}
			for (int i = 0; i < samples; ++i){
				VectorXd W;
				if (simulation){
					W = this->weights.row(i);
				}
				else{
					W = this->weights.row(this->weights.rows()-1-i);	
				}

				VectorXd temp = W.tail(W.rows()-1);
				this->logistic_regression.setWeights(temp);
				this->logistic_regression.setBias(W(0));
				temp_predict.row(i) = this->logistic_regression.predict(X_test, prob, data_processing);
				data_processing = false;
			}

			predict = temp_predict.colwise().mean();
			if (!prob){
				predict.noalias() = predict.unaryExpr([](double elem){
					return (elem > 0.5) ? 1.0 : 0.0;
				});
			}
		}
		else{
			VectorXd W = this->mean_weights;
			VectorXd temp = W.tail(W.rows()-1);
			this->logistic_regression.setWeights(temp);
			this->logistic_regression.setBias(W(0));
			predict = this->logistic_regression.predict(X_test, prob, data_processing);
		}
		
		return predict;
		
	}
	else{
		cout << "Error: No initialized function"<< endl;
		return predict;
	}
}

void GPU_Hamiltonian_MC::getModel(VectorXd& weights, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin, double& bias){
	weights = this->mean_weights.tail(this->mean_weights.rows()-1);
	bias = this->mean_weights(0);
	featureMean = this->logistic_regression.featureMean;
	featureStd = this->logistic_regression.featureStd;
	featureMax = this->logistic_regression.featureMax;
	featureMin = this->logistic_regression.featureMin;
}

void GPU_Hamiltonian_MC::loadModel(VectorXd weights, VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias){
	this->logistic_regression.init(this->normalization, this->standarization, true);
	this->logistic_regression.setWeights(weights);
	this->logistic_regression.setBias(bias);
	this->logistic_regression.featureMean = featureMean;
	this->logistic_regression.featureStd = featureStd;
	this->logistic_regression.featureMax = featureMax;
	this->logistic_regression.featureMin = featureMin;
	VectorXd temp(weights.rows() +1 );
	temp << bias, weights;
	this->mean_weights = temp;
}
void GPU_Hamiltonian_MC::setData(MatrixXd &_X,VectorXd &_Y){
	this->logistic_regression.setData(_X, _Y);
}



