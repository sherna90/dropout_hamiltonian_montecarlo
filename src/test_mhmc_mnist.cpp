//#include "hmc.hpp"
#include "utils/c_utils.hpp"

#include <iostream>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>
#include <chrono>
#include <ctime>
#include <random>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "likelihood/CPU_mhmc.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{
	
	
	C_utils utils;
	
			
	MatrixXd X_train;
	MatrixXd X_test;
	VectorXd Y_train;
	VectorXd Y_test;
	utils.read_Data("../data/MNIST/X_train.csv",X_train);
	utils.read_Data("../data/MNIST/X_test.csv",X_test);
	utils.read_Labels("../data/MNIST/Y_train.csv",Y_train);
	utils.read_Labels("../data/MNIST/Y_test.csv",Y_test);

	//RowVectorXd mean = X_train.colwise().mean();
	//X_train = X_train.rowwise() - mean;
	//X_test =X_test.rowwise() - mean;
	
	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	double lambda = 1.0;
	int epochs = 100;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = iterations*0.1;
	double step_size = 1e-4;
	int num_steps = 1e2;
	int samples = 1000;
	int psamples = 30;

	cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Samples: "<< samples << ", Step Size: " << step_size << ", Num Steps:" << num_steps << endl;

	CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, false, false, samples);

	cout << "Init run" << endl;
	hmc.run();

	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, psamples, false);

	cout << "Prob" << endl;
	MatrixXd predict_proba = hmc.get_predict_proba();
	utils.writeToCSVfile("predict_proba_mean.csv", predict_proba);
	utils.writeToCSVfile("predict_proba_max.csv", (predict_proba.rowwise().maxCoeff()));
	MatrixXd predict_proba_std = hmc.get_predict_proba_std();
	utils.writeToCSVfile("predict_proba_std.csv", predict_proba_std);
	utils.writeToCSVfile("Y_test.csv", Y_test);
	cout << predict_proba << endl;
	cout << "Mean Prob" << endl;
	VectorXd mean_prob = predict_proba.colwise().mean();
	cout << mean_prob.transpose() << endl;
	cout << "Std Prob" << endl;
	VectorXd std_prob = ((predict_proba.rowwise() - mean_prob.transpose()).array().square().colwise().sum() / (predict_proba.rows())).sqrt();
	cout << std_prob.transpose() << endl;

	MatrixXd predict_history = hmc.get_predict_history();
	VectorXd histogram(psamples);
	for (int i = 0; i < predict_history.cols(); ++i){
		histogram(i) = utils.calculateAccuracyPercent(Y_test, predict_history.col(i));
	}
	utils.writeToCSVfile("histogram.csv", histogram);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	return 0;
}
