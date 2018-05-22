//Author: Diego Vergara
#ifndef Mask_CPU_HAMILTONIAN_MC_H
#define Mask_CPU_HAMILTONIAN_MC_H
#include "CPU_logistic_regression.hpp"
#include "hmc.hpp"

class Mask_CPU_Hamiltonian_MC : public Hamiltonian_MC
{
public:
	void init( MatrixXd &_X, VectorXd &_Y, double _lambda = 1.0, int _warmup_iterations = 100, int _iterations = 1000, double _step_size = 0.01, int _num_step = 100, bool _normalization =true, bool _standarization=true, double _path_lenght = 0.0);
	void run(bool warmup_flag = false, bool for_predict = false);
	VectorXd predict(MatrixXd &_X_test, bool prob = false, int samples = 1, bool simulation = false);
	MatrixXd get_maskMatrix();
	void set_maskMatrix(MatrixXd &_mask_matrix);
	void getModel(VectorXd& weights, VectorXd& featureMean, VectorXd& featureStd, VectorXd& featureMax, VectorXd& featureMin, double& bias);
	void loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias);
	VectorXd gradient(VectorXd &W,VectorXd &mask);
	//VectorXd gradient(VectorXd &W);
	double logPosterior(VectorXd &W, VectorXd &mask, bool precompute = true);
	//double logPosterior(VectorXd &W, bool precompute = true);
	//VectorXd concrete_dropout(VectorXd x, double p);
	//double regularizer(VectorXd x, double p, double weights_regularizer, double mask_rate);
	void setData(MatrixXd &_X,VectorXd &_Y, bool _preprocesing = true);
protected:
 	MatrixXd mask_matrix;
 	bool normalization, standarization;
 	CPU_LogisticRegression logistic_regression;
 	double mask_rate;
};

#endif // Mask_CPU_HAMILTONIAN_MC_H