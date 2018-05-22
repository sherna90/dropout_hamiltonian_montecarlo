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
#include "likelihood/CPU_hmc_dropout.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{
    C_utils utils;
    if(argc != 23) {
        cerr <<"Incorrect input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    else{
        string train_csv_path, test_csv_path;
        if(strcmp(argv[1], "-train") == 0) {
            train_csv_path=argv[2];
        }
        else{
            cerr <<"No train path given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        if(strcmp(argv[3], "-test") == 0) {
            test_csv_path=argv[4];
        }
        else{
            cerr <<"No test path given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        double lambda;
        if(strcmp(argv[5], "-l") == 0) {
            lambda=atof(argv[6]);
        }
        else{
            cerr <<"No lambda parameter given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        int warmup;
        if(strcmp(argv[7], "-w") == 0) {
            warmup=atoi(argv[8]);
        }
        else{
            cerr <<"No warmup iteration number given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        int iterations;
        if(strcmp(argv[9], "-nit") == 0) {
            iterations=atoi(argv[10]);
        }
        else{
            cerr <<"No hmc iteration number given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        double step_size;
        if(strcmp(argv[11], "-sz") == 0) {
            step_size=atof(argv[12]);
        }
        else{
            cerr <<"No hmc step size given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        int num_steps;
        if(strcmp(argv[13], "-ns") == 0) {
            num_steps=atoi(argv[14]);
        }
        else{
            cerr <<"No hmc number of steps given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        int num_pred;
        if(strcmp(argv[15], "-np") == 0) {
            num_pred=atoi(argv[16]);
        }
        else{
            cerr <<"No hmc number of prediction given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
        bool normalization;
        if(strcmp(argv[17], "-n") == 0) {
            stringstream ss(argv[18]);
            if(!(ss >> boolalpha >> normalization)) {
                cerr <<"hmc normalization mode imput error" << endl;
                cerr <<"exiting..." << endl;
                return EXIT_FAILURE;
            }
        }
        bool standarization;
        if(strcmp(argv[19], "-s") == 0) {
            stringstream ss(argv[20]);
            if(!(ss >> boolalpha >> standarization)) {
                cerr <<"hmc standarization mode imput error" << endl;
                cerr <<"exiting..." << endl;
                return EXIT_FAILURE;
            }
        }
       	bool mask = true;
        double mask_rate;
        if(strcmp(argv[21], "-mr") == 0) {
            mask_rate=atof(argv[22]);
        }
        else{
            cerr <<"No hmc Dropout mask rate given" << endl;
            cerr <<"exiting..." << endl;
            return EXIT_FAILURE;
        }
		

        cout << "Lambda: " << lambda << ", Warmup: " << warmup << ", Iterations: " << iterations << ", Step Size: " << step_size << ", Num Steps:" << num_steps << ", Normalization: " << normalization <<", Standarization: " << standarization <<", Num Pred: " << num_pred << ", Mask: " << mask << ", Mask_rate: " << mask_rate <<endl;
        
		MatrixXd train, test;

	  	/*train_csv_path = "../data/gisette_scale.csv";
	  	test_csv_path = "../data/gisette_scale_t.csv";*/

		utils.read_Data(train_csv_path,train);
		utils.read_Data(test_csv_path,test);
		
		MatrixXd X_train = train.block(0,1,train.rows(),train.cols()-1);
		MatrixXd X_test = test.block(0,1,test.rows(),test.cols()-1);
		VectorXd Y_train = train.block(0,0,train.rows(),1);
		VectorXd Y_test = test.block(0,0,test.rows(),1);

		for (int i = 0; i < Y_train.rows(); ++i){
			if(Y_train(i) == -1){
				Y_train(i) = 0;
			}
		}

		for (int i = 0; i < Y_test.rows(); ++i){
			if(Y_test(i) == -1){
				Y_test(i) = 0;
			}
		}

		VectorXd predicted_labels;

		/*double lambda = 1.0;
		int warmup = 1e2;
		int iterations = 1e3;
		double step_size = 1e-2;
		int num_steps = 1e2;
		bool mask = true;
		double mask_rate = 0.5;*/

		clock_t start;
		double duration;
	   	start = std::clock();

        int minibatch = 100;  // Agregar
        int samples = 100; // Agregar

        Mask_CPU_Hamiltonian_MC hmc;
        hmc.init(X_train, Y_train, lambda, warmup, iterations, minibatch, step_size, num_steps, mask_rate, normalization, standarization, samples);
		
        hmc.run();

        predicted_labels = hmc.predict(X_test num_pred);

        utils.confusion_matrix(Y_test, predicted_labels, true);
        utils.report(Y_test, predicted_labels, true);
        utils.calculateAccuracyPercent(Y_test, predicted_labels);
        
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        cout << "Elapsed Time: " << duration << " s" << endl;
        cout << "=====================================================================" << endl;
		
	}
	return 0;
}
