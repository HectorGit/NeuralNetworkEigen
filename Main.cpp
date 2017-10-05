/*
  ==============================================================================

    Main.cpp
    Created: 29 Sep 2017 3:10:51pm
    Author:  Owner

  ==============================================================================
*/
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <iostream>
#include "NeuralNetwork.h"

int main(){

	std::vector<int> paramSizes = { 2,3,2 };
	NeuralNetwork* aNet = new NeuralNetwork(paramSizes);

	int mini_batch_size = 10;

	Eigen::MatrixXf yTemp = Eigen::MatrixXf::Zero(2, 1); //10 by 1 just one number - a classification
	yTemp(0, 0) = 0;
	yTemp(1, 0) = 1;

	Eigen::MatrixXf yTempNeg = Eigen::MatrixXf::Zero(2, 1); //10 by 1 just one number - a classification
	yTemp(0, 0) = 1;
	yTemp(1, 0) = 0;

	for (int i = 0; i < mini_batch_size; i++) {
		Eigen::MatrixXf temp = Eigen::MatrixXf::Random(2, 1);
		aNet-> all_Xs.emplace_back(temp);
		if (i % 2 == 0) {
			aNet->all_Ys.emplace_back(yTemp);
		}
		else {
			aNet->all_Ys.emplace_back(yTempNeg);
		}
	}

	//should backprop take y as parameter
	aNet->backPropagation(mini_batch_size); //mini_batch_size is 10;

	/*SGD(training_data, epochs, mini_batch_size, eta,test_data)*/
	//myNetwork.SGD(trainData, 30, 10, 3.0, testData); //30 epochs, //10 vectors for mini batch, 3.0 error, 
	return 0;
}

/*
	UPDATE:

	DONE: (lack dynamism - might improve on this)
		1) void feedForward(Eigen::MatrixXf &x);
		2) int evaluate(Eigen::MatrixXf &activationL2 , Eigen::MatrixXi &y);
		3) Eigen::MatrixXf costDerivative(Eigen::MatrixXf &outputActivations, Eigen::MatrixXf &y);
		4) Eigen::MatrixXf sigmoid_Vectorial(Eigen::MatrixXf &z);
		5) Eigen::MatrixXf sigmoid_Prime_Vectorial(Eigen::MatrixXf &z);

	PENDING:
		stochasticGradientDescent
		updateMiniBatch
		backPropagation
		backPropagationForWs
		backPropagationForBs

*/




