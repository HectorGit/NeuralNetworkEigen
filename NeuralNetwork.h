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

	//make these private again later

	Eigen::MatrixXf activationL1;
	Eigen::MatrixXf activationL2;

	Eigen::MatrixXf x; //includes all the input vectors
	Eigen::MatrixXf y; //includes all the output vectors //or the classification result. e.g. appropriate index

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

private:
	int numLayers;
	std::vector<int> sizes; // is this what I want?
	float eta;

	//GLOBAL - check which ones we need...
	Eigen::MatrixXf testData;
	Eigen::MatrixXf validationData;
	Eigen::MatrixXf trainingData;

	//make those things private again!!


	//=================================================================================
		

	//void stochasticGradientDescent(Eigen::MatrixXf &trainingData, int epochs, int miniBatchSize, float eta, Eigen::MatrixXf &testData);
	//void updateMiniBatch(Eigen::MatrixXf &mini_batch, float eta, Eigen::MatrixXf &biasesMatrixL1, Eigen::MatrixXf &weightsMatrixL1, Eigen::MatrixXf &biasesMatrixL2, Eigen::MatrixXf &weightsMatrixL2);
	//void backPropagation(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2); //input vs labels
	//void backPropagationForWs(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2);
	//void backPropagationForBs(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2);

	void feedForward(Eigen::MatrixXf &x/*Eigen::MatrixXf &activationL1, Eigen::MatrixXf &activationL2*/);
	int evaluate(Eigen::MatrixXf &activationL2 /*Eigen::MatrixXf &testData*/, Eigen::MatrixXi &y);
	Eigen::MatrixXf costDerivative(Eigen::MatrixXf &outputActivations, Eigen::MatrixXf &y);
	Eigen::MatrixXf sigmoid_Vectorial(Eigen::MatrixXf &z);
	Eigen::MatrixXf sigmoid_Prime_Vectorial(Eigen::MatrixXf &z);

};			