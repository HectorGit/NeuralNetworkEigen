/*
  ==============================================================================

    NeuralNetwork.h
    Created: 29 Sep 2017 2:54:03pm
    Author:  Owner

  ==============================================================================
*/

#pragma once

#include <Eigen/Core>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

class NeuralNetwork {

public:
	NeuralNetwork(std::vector<int> &paramSizes);
	~NeuralNetwork();

private:
	int numLayers;
	Eigen::RowVector2i sizes; // is this what I want?
	float eta;

	Eigen::Matrix2f biasesMatrix;//init random
	Eigen::Matrix2f weightsMatrix;//init random
	
	Eigen::Matrix2f gradient_Ws;
	Eigen::Matrix2f gradient_Bs;

	//=================================================================================
	
	//***question- what does zip do in python?
	
	std::vector<float> feedForward(Eigen::Matrix2f activation);

	void stochasticGradientDescent(Eigen::Matrix2f trainingData, int epochs, int miniBatchSize, float eta, Eigen::Matrix2f testData);

	void updateMiniBatch(Eigen::Matrix2f mini_batch, float eta, Eigen::Matrix2f biasesMatrix, Eigen::Matrix2f weightsMatrix);

	/* split this into two methods - will have to repeat some work (no multiple return on c++)*/
	/*PASSING IN THE GRADIENT MATRICES*/ // ALTHOUGH THEY ARE MEMBERS SO MIGHT NOT NEED TO, BUT FOR CLARITY.
	void backPropagation(Eigen::Matrix2f x, Eigen::Matrix2f y, Eigen::Matrix2f gradient_Ws, Eigen::Matrix2f gradient_Bs); //input vs labels
	void backPropagationForWs(Eigen::Matrix2f x, Eigen::Matrix2f y, Eigen::Matrix2f gradient_Ws, Eigen::Matrix2f gradient_Bs));
	void backPropagationForBs(Eigen::Matrix2f x, Eigen::Matrix2f y, Eigen::Matrix2f gradient_Ws, Eigen::Matrix2f gradient_Bs));

	Eigen::RowVector2f evaluate(Eigen::Matrix2f testData); //could also just be ONE number ?!

	Eigen::RowVector2f cost_derivative(Eigen::Matrix2f outputActivations, Eigen::Matrix2f y);

	Eigen::RowVector2f sigmoid_Vectorial(Eigen::Matrix2f z);

	Eigen::RowVector2f sigmoid_prime_Vectorial(Eigen::Matrix2f z);

};			