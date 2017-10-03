/*
  ==============================================================================

    NeuralNetwork.h
    Created: 29 Sep 2017 2:54:03pm
    Author:  Owner

  ==============================================================================
*/

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>

class NeuralNetwork {

public:
	NeuralNetwork(std::vector<int> &paramSizes);
	~NeuralNetwork();

private:
	int numLayers;
	std::vector<int> sizes; // is this what I want?
	float eta;

	//LEVEL 1
	Eigen::MatrixXf weightsMatrixL1;//init random
	Eigen::MatrixXf gradientWsL1;
	Eigen::MatrixXf biasesMatrixL1;
	Eigen::MatrixXf gradientBsL1;

	//LEVEL 2
	Eigen::MatrixXf weightsMatrixL2;//init random
	Eigen::MatrixXf gradientWsL2;
	Eigen::MatrixXf biasesMatrixL2;
	Eigen::MatrixXf gradientBsL2;

	//=================================================================================
		
	//std::vector<float> feedForward(Eigen::MatrixXf &activation);

	//void stochasticGradientDescent(Eigen::MatrixXf &trainingData, int epochs, int miniBatchSize, float eta, Eigen::MatrixXf &testData);

	//void updateMiniBatch(Eigen::MatrixXf &mini_batch, float eta, Eigen::MatrixXf &biasesMatrixL1, Eigen::MatrixXf &weightsMatrixL1, Eigen::MatrixXf &biasesMatrixL2, Eigen::MatrixXf &weightsMatrixL2);

	/* split this into two methods - will have to repeat some work (no multiple return on c++)*/
	/*PASSING IN THE GRADIENT MATRICES*/ // ALTHOUGH THEY ARE MEMBERS SO MIGHT NOT NEED TO, BUT FOR CLARITY.
	//void backPropagation(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2); //input vs labels
	//void backPropagationForWs(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2);
	//void backPropagationForBs(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2);

	//Eigen::RowVectorXf evaluate(Eigen::MatrixXf testData); //could also just be ONE number ?!

	//Eigen::RowVectorXf cost_derivative(Eigen::MatrixXf outputActivations, Eigen::MatrixXf y);

	//Eigen::RowVectorXf sigmoid_Vectorial(Eigen::MatrixXf z);

	//Eigen::RowVectorXf sigmoid_prime_Vectorial(Eigen::MatriXf z);

};			