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
#include "likelihood/CPU_hmc.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{
	C_utils utils;
	
	
	string data_csv_path, labels_csv_path;
	MatrixXd iris;

  	data_csv_path = "../data/iris.csv";

	utils.read_Data(data_csv_path,iris);
	// Sepal Length, Sepal Width, Petal Length and Petal Width
    // Setosa, Versicolour, and Virginica
	MatrixXd data_train(100, 2);
	data_train << iris.block(0,0,100,1) , iris.block(0,1,100,1);//, iris.block(0,2,150,1), iris.block(0,3,150,1);
	VectorXd labels_train=iris.block(0,4,100,1);
	/*MatrixXd train=iris.block(0,0,150,4);
	VectorXd l_train=iris.block(0,4,150,1);
	MatrixXd X_train, X_test;
	VectorXd Y_train, Y_test;
	utils.dataPermutation(train, l_train);
	utils.dataPartition(train, l_train, X_train, X_test, Y_train, Y_test, 50);
	utils.writeToCSVfile("X_train.csv", X_train);
	utils.writeToCSVfile("X_test.csv", X_test);
	utils.writeToCSVfile("Y_train.csv", Y_train);
	utils.writeToCSVfile("Y_test.csv", Y_test);*/
	/*for (int i = 0; i < labels_train.rows(); ++i){
		if(labels_train(i) == 2){
			labels_train(i) = 0;
		}
	}*/
	
	MatrixXd X_train = data_train;
	VectorXd Y_train = labels_train;

	MatrixXd X_test = data_train;
	VectorXd Y_test = labels_train;

	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	double lambda = 0.1;
	int epochs = 10;
	int mini_batch=100;
	int num_batches=X_train.rows()/mini_batch;
	int iterations = epochs*num_batches;
	int warmup = iterations*0.1;
	double step_size = 1e-2;
	int num_steps = 1e2;
	int samples = 100;

	cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Mini Batch: "<< mini_batch <<", Samples: " << samples << ", Step Size: " << step_size << ", Num Steps:" << num_steps << endl;

	CPU_Hamiltonian_MC hmc;
	hmc.init(X_train, Y_train, lambda, warmup, iterations, mini_batch, step_size, num_steps, false, true, samples);

	cout << "Init run" << endl;
	hmc.run();
	MatrixXd weights = hmc.get_weights();
	utils.writeToCSVfile("hmc_weights.csv", weights);
	cout << "Init predict" << endl;
	predicted_labels = hmc.predict(X_test, 10, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	utils.confusion_matrix(Y_test, predicted_labels, true);
	
	return 0;
}
