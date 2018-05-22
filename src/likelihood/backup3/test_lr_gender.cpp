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
#include "likelihood/CPU_logistic_regression.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{

	C_utils utils;
		
	string data_csv_path, labels_csv_path;
	MatrixXd data;
	VectorXd labels;

	int train_partition = 14000;
	data_csv_path = "../data/dataset.csv";
	labels_csv_path = "../data/gender_label.csv";

	cout << "Read Data" << endl;

	utils.read_Data(data_csv_path,data);
	utils.read_Labels(labels_csv_path,labels);

	/*int cols = utils.get_Cols(data_csv_path, ',');
	utils.read_Data(data_csv_path,data, 2000, cols);
	utils.read_Labels(labels_csv_path,labels, 2000);	
	int train_partition = 1600;*/

	//cout << "Data Permutation" << endl;
	//utils.dataPermutation(data, labels);

	cout << "Data Partition" << endl;
	MatrixXd data_train, data_test;
	VectorXd labels_train, labels_test;
	utils.dataPartition(data, labels, data_train, data_test, labels_train, labels_test, train_partition);

	cout << "Init" << endl;
	VectorXd predicted_labels;
	
	double lambda = 1.0;
	int iterations = 1e3;
	double alpha = 0.99;
	int tol = 1;

	cout << "Lambda: " << lambda <<  ", Iterations: " << iterations << ", Alpha: " << alpha << ", Tolerance:" << tol << endl;
	MatrixXd X_train = data_train;
	MatrixXd X_test = data_test;
	VectorXd Y_train = labels_train;
	VectorXd Y_test = labels_test;

	CPU_LogisticRegression lr;
	lr.init(X_train, Y_train, lambda, true, true, true);

	cout << "Init train" << endl;
	lr.train(iterations, alpha, tol);
	cout << "Init predict" << endl;
	predicted_labels = lr.predict(X_test, false);

	cout << "Init report" << endl;
	utils.report(Y_test, predicted_labels, true);
	utils.calculateAccuracyPercent(Y_test, predicted_labels);
	utils.confusion_matrix(Y_test, predicted_labels, true);

	return 0;
}
