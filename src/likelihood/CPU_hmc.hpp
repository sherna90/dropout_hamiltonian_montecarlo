//Author: Diego Vergara
#ifndef CPU_HAMILTONIAN_MC_H
#define CPU_HAMILTONIAN_MC_H
#include "CPU_logistic_regression.hpp"
#include "hmc.hpp"

class CPU_Hamiltonian_MC : public Hamiltonian_MC
{
public:
	void init( MatrixXd &_X, VectorXd &_Y, double _lambda = 1.0, int _warmup_iterations = 100, int _iterations = 1000, int _minibatch = 100, double _step_size = 0.01, int _num_step = 100, bool _normalization =true, bool _standarization=true, int _samples = 1000, double _path_lenght = 0.0);
	void run(bool warmup_flag = false, bool for_predict = false, double mom = 0.99);
	VectorXd predict(MatrixXd &_X_test, int psamples = 1, bool simulation = false, bool data_processing = true);
	void getModel(MatrixXd& weights, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin);
	void loadModel(MatrixXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, bool _normalization = true, bool _standarization =true);
	VectorXd gradient(VectorXd &W, int n_iter);
	double logPosterior(VectorXd &W, int n_iter, bool precompute = true);
	void setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing = true);
	void saveModel(string name);
protected:
 	CPU_LogisticRegression logistic_regression;
 	bool normalization, standarization;
};

#endif // CPU_HAMILTONIAN_MC_H