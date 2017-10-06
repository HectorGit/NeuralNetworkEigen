/*
  ==============================================================================

    NeuralNetwork.cpp
    Created: 29 Sep 2017 2:54:03pm
    Author:  Owner

	//VERSION 1 OF CODE BASED ON ALEX THOMO'S IMPLEMENTATION

  ==============================================================================
*/

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <math.h> //or cmath?
#include "NeuralNetwork.h"

/*
NOTES: MAKE IT MORE EXPLICIT WHAT EACH FUNCTION IS TAKING IN AND RETURNING IF YOU CAN?!!
*/

//Constructor
NeuralNetwork::NeuralNetwork(std::vector<int> &paramSizes)
{
	bool debug = false;
	//printf("what is going on?");
	numLayers = paramSizes.size();
	mini_batch_size = 10;
	if (debug) { printf("\n numLayers is: %d \n", numLayers); }

	eta = 3.0; //should pass this in to SGD --> CHECK... DONT SET IT HERE!
	sizes = paramSizes; //NOT SURE IF THIS WORKS //MIGHT NEED TO COPY IT IN SOMEHOW.

	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) { printf("\n------------Initializing NeuralNet------------\n"); }

	if (debug) { printf("\nLEVEL ONE\n"); }

	weightsMatrixL1 = Eigen::MatrixXf::Random(sizes[1], sizes[0]); //check this
	if (debug) {printf("\nHere's weightsMatrixL1\n");
		std::cout << weightsMatrixL1.format(CleanFmt);}
	gradientWsL1 = Eigen::MatrixXf::Zero(sizes[1], sizes[0]);
	if (debug) {printf("\nHere's gradientWsL1\n");
		std::cout << gradientWsL1.format(CleanFmt);}

	biasesMatrixL1 = Eigen::MatrixXf::Random(sizes[1], 1);
	if (debug) {printf("\nHere's biasesMatrixL1\n");
		std::cout << biasesMatrixL1.format(CleanFmt);}
	gradientBsL1 = Eigen::MatrixXf::Zero(sizes[1], 1);
	if (debug) {printf("\nHere's gradientBsL1\n");
		std::cout << gradientBsL1.format(CleanFmt);}

	if (debug) { printf("\n\nLEVEL TWO\n"); }

	weightsMatrixL2 = Eigen::MatrixXf::Random(sizes[2], sizes[1]);
	if (debug) {printf("\nHere's weightsMatrixL2\n");
		std::cout << weightsMatrixL2.format(CleanFmt);}
	gradientWsL2 = Eigen::MatrixXf::Zero(sizes[2], sizes[1]);
	if (debug) {printf("\nHere's gradientWsL2\n");
		std::cout << gradientWsL2.format(CleanFmt);}

	biasesMatrixL2 = Eigen::MatrixXf::Random(sizes[2], 1);
	if (debug) {printf("\nHere's biasesMatrixL2\n");
		std::cout << biasesMatrixL2.format(CleanFmt);}
	gradientBsL2 = Eigen::MatrixXf::Zero(sizes[2], 1);
	if (debug) {printf("\nHere's gradientBsL2\n");
		std::cout << gradientBsL2.format(CleanFmt);}

	if (debug) { printf("\n --------DONE Initializing NeuralNet------- \n"); }

}

NeuralNetwork::~NeuralNetwork() 
{
	//release all memory or something Lolz
}	

void NeuralNetwork::feedForward(Eigen::MatrixXf &x, Eigen::MatrixXf &y)
{
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) {
		printf("--------feedforward algorithm------");
		printf("\n----------LEVEL ONE ---------\n");
	}

	Eigen::MatrixXf dotProductL1 = weightsMatrixL1 * x;
	if (debug) {
		printf("\n wT dot x \n");
		std::cout << "\n w: \n" << weightsMatrixL1.format(CleanFmt);
		std::cout << "\n x: \n" << x.format(CleanFmt);
		std::cout << "\n Result: \n" << dotProductL1.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	zL1 = dotProductL1 + biasesMatrixL1;
	if (debug) {
		printf("\n + Biases \n");
		std::cout << "\n b: \n" << biasesMatrixL1.format(CleanFmt);
		std::cout << "\n Result: \n" << zL1.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	activationL1 = sigmoid_Vectorial(zL1);
	if (debug) {
		printf("\n\n Apply Sigmoid\n");
		std::cout << "\n Result: \n" << activationL1.format(CleanFmt);
	}

	if (debug) { printf("\n\n---------LEVEL TWO ----------\n"); }

	Eigen::MatrixXf dotProductL2 = weightsMatrixL2*activationL1;
	if (debug) {
		printf("\n wT dot activationL1 \n");
		std::cout << "\n w: \n" << weightsMatrixL2.format(CleanFmt);
		std::cout << "\n activationL1: \n" << activationL1.format(CleanFmt);
		std::cout << "\n Result: \n" << dotProductL2.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	zL2 = dotProductL2 + biasesMatrixL2;
	if (debug) {
		printf("\n + Biases \n");
		std::cout << "\n b: \n" << biasesMatrixL2.format(CleanFmt);
		std::cout << "\n Result: \n" << zL2.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	activationL2 = sigmoid_Vectorial(zL2);
	if (debug) {
		printf("\n\n Apply Sigmoid\n");
		std::cout << "\n Result: \n" << activationL2.format(CleanFmt);
	}
	if (debug) { printf("\n\n---------DONE----------\n"); }

}

/*Requires FeedForward to be working*/ /*called after an 'epoch' ends*/
									   /*do we really need testData here??*/
int NeuralNetwork::evaluate(Eigen::MatrixXf &activationL2/*Eigen::MatrixXf &testData*/, Eigen::MatrixXi &y)
{
	/*
	in this implementation it just acts as evaluating
	a single row of outputs w float values versus a int-valued 1D matrix
	fix when the time comes.
	*/

	bool debug = false;

	int counterOfMatches = 0;

	Eigen::MatrixXi::Index classificationIndex;
	Eigen::MatrixXf maxVal = Eigen::MatrixXf::Zero(10, 1);

	for (int m = 0; m< activationL2.rows(); m++)
	{
		maxVal(m) = activationL2.row(m).maxCoeff(&classificationIndex);
		if (debug) { printf("\n comparing %d, %d \n", classificationIndex, y(m)); }
		if (classificationIndex == y(m))
		{
			if (debug) { printf("\n match %d, %d \n", classificationIndex, y(m)); }
			counterOfMatches++;
		}
	}
	return counterOfMatches;
}

Eigen::MatrixXf NeuralNetwork::costDerivative(Eigen::MatrixXf &outputActivations, Eigen::MatrixXf &y)
{
	//should this be in a for loop for every instance that came in?? or is it called multiple times?
	return outputActivations - y;
}

Eigen::MatrixXf NeuralNetwork::sigmoid_Vectorial(Eigen::MatrixXf &z)
{
	int r = z.rows();
	int c = z.cols();
	Eigen::MatrixXf returnedMatrix = Eigen::MatrixXf::Zero(r, c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			returnedMatrix.array()(i, j) = 1.0 / (1.0 + exp(-1.0 * z.array()(i, j))); //exp comes from Math.
		}
	}
	return returnedMatrix;
}

Eigen::MatrixXf NeuralNetwork::sigmoid_Prime_Vectorial(Eigen::MatrixXf &z)
{
	int r = z.rows();
	int c = z.cols();
	Eigen::MatrixXf sigmoidMatrix = sigmoid_Vectorial(z);
	Eigen::MatrixXf returnedMatrix = Eigen::MatrixXf::Zero(r, c);
	returnedMatrix = sigmoidMatrix.array()*(1 - sigmoidMatrix.array());
	return returnedMatrix;
}

void NeuralNetwork::stochasticGradientDescent(/*Eigen::MatrixXf &trainingData, int epochs, int miniBatchSize, float eta, Eigen::MatrixXf &testData*/)
{	
	/*
	implement a for loop to select mini batches OR
	it looks like we could put this into a for loop 
	which grabs all_Xs[] from k to k+miniBatchSize.
	it also grabs corresponding all_Ys[] from k to k_miniBatchSize.
	it will check if there are still .

	*/
	bool debug = true;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");


	if (debug) { printf("\n stochastic gradient descent was called \n"); }

	int numberOfMiniBatches = all_Xs.size()/mini_batch_size;
	if (debug) { printf("\n numberofMiniBatches, %d \n", numberOfMiniBatches); }
	
	int k = mini_batch_size;
	epochs = numberOfMiniBatches;

	for (int i = 0; i < epochs; i++) {

		//before running for that epoch,
		//we select the mini batches and put them
		//in mini_Batch_Xs and mini_Batch_Ys
		for (int m = i*k; m < (i + 1)*k; m++) {
			mini_Batch_Xs.emplace_back(all_Xs[m]); 
			mini_Batch_Ys.emplace_back(all_Ys[m]);
		}

		if (debug) {
			printf("\n --------MINI BATCH Xs %d -------- \n", i);
			for (int j = 0; j < k; j++) {
				std::cout << "\n" << mini_Batch_Xs[j].format(CleanFmt) << "\n";
				printf("\n");
			}
			printf("\n --------MINI BATCH Ys %d -------- \n", i);
			for (int j = 0; j < k; j++) {
				std::cout << "\n" << mini_Batch_Ys[j].format(CleanFmt) << "\n";
				printf("\n");
			}
		}

		printf("\n epoch [%d] \n", i);
		updateMiniBatch();

	}
}

void NeuralNetwork::updateMiniBatch(/*Eigen::MatrixXf &mini_batch,*/ /*float eta*//*Eigen::MatrixXf &biasesMatrixL1,*/ /*Eigen::MatrixXf &weightsMatrixL1, Eigen::MatrixXf &biasesMatrixL2, Eigen::MatrixXf &weightsMatrixL2*/)
{
	/*we have defined the following matrices already*/
	/*LEVEL 1
	Eigen::MatrixXf weightsMatrixL1; 3 by 2 filled w current ws
	Eigen::MatrixXf gradientWsL1;    3 by 2 filled with zeros now
	Eigen::MatrixXf biasesMatrixL1;  3 by 1 filled w current bs
	Eigen::MatrixXf gradientBsL1;    3 by 1 filled with zeros now*/

	/*LEVEL 2
	Eigen::MatrixXf weightsMatrixL2; 2 by 3  filled w current ws
	Eigen::MatrixXf gradientWsL2;    2 by 3  filled with zeros now
	Eigen::MatrixXf biasesMatrixL2;  2 by 1  filled w current bs
	Eigen::MatrixXf gradientBsL2;    2 by 1  filled with zeros now */

	bool debug = true;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) { printf("\n update mini batch was called \n"); }

	if (debug) {
		printf("\n old weights and biases \n");
		std::cout << "\n weightsMatrixL1 \n" << weightsMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n weightsMatrixL2 \n" << weightsMatrixL2.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL1 \n" << biasesMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL2 \n" << biasesMatrixL2.format(CleanFmt) << "\n";
	}

	
	//need to move what is in main to here for the moment. (DUMMY STUFF)
	backPropagation(mini_batch_size);
	
	//1. accumulate all the gradients.
	for (int i = 0; i < mini_batch_size; i++) {
		/*if (debug) {
			printf("\n gradientWsL1 rows = %d cols = %d \n", gradientWsL1.rows(), gradientWsL1.cols());
			std::cout << "\n" << gradientWsL1.format(CleanFmt) << "\n";

			printf("\n allGradientsWs[%d][0] rows = %d cols = %d \n", i, allGradientsWs[i][0].rows(), allGradientsWs[i][0].cols());
			std::cout << "\n" << allGradientsWs[i][0].format(CleanFmt) << "\n";
		}*/

		gradientWsL1 = gradientWsL1 + allGradientsWs[i][0];
		gradientWsL2 = gradientWsL2 + allGradientsWs[i][1];
		gradientBsL1 = gradientBsL1 + allGradientsBs[i][0];
		gradientBsL2 = gradientBsL2 + allGradientsBs[i][1];
	}

	
	//2. get an average (+other operations)
	printf("\n value of the multiplier for gradient matrices = %4.2f ", eta / (float) mini_batch_size);
	weightsMatrixL1 = weightsMatrixL1 - ((eta / (float)mini_batch_size) * gradientWsL1); 
	weightsMatrixL2 = weightsMatrixL2 - ((eta / (float)mini_batch_size) * gradientWsL2);
	biasesMatrixL1 = biasesMatrixL1 - ((eta / (float)mini_batch_size) * gradientBsL1);
	biasesMatrixL2 = biasesMatrixL2 - ((eta / (float)mini_batch_size) * gradientBsL2);
	

	if (debug) {
		printf("\n new weights and biases \n");
		std::cout << "\n weightsMatrixL1 \n" << weightsMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n weightsMatrixL2 \n" << weightsMatrixL2.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL1 \n" << biasesMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL2 \n" << biasesMatrixL2.format(CleanFmt) << "\n";
	}
}

void NeuralNetwork::backPropagation(int mini_batch_size/*, mini_Batch_Xs, mini_Batch_Ys*/)
{	
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	/*
	these became class members.
	Eigen::MatrixXf allGradientsBs[10][2]; //nabla_b
	Eigen::MatrixXf allGradientsWs[10][2]; //nabla_w
	Eigen::MatrixXf allActivations[10][3];
	Eigen::MatrixXf allZs[10][2];
	*/

	//STEP 1 - 
	for (int i = 0; i < mini_batch_size; i++) {
		if (debug){ printf("\n feedforward for mini_Batch_Xs[%d] \n", i); }
		feedForward(mini_Batch_Xs[i], mini_Batch_Ys[i]); //does this work?
		
		allActivations[i][0] = mini_Batch_Xs[i];
		allActivations[i][1] = activationL1;
		allActivations[i][2] = activationL2;

		allZs[i][0] = zL1;
		allZs[i][1] = zL2;
	}

	if (debug) {
		printf("\n starting \n");
		for (int i = 0; i < mini_batch_size; i++) {
		
			printf("\n showing allActivations for single_x # %d \n", i);
			for (int k = 0; k < numLayers; k++) {
				Eigen::MatrixXf temp = allActivations[i][k];
				std::cout << "\n" << temp.format(CleanFmt) << "\n";
			}

			printf("\n showing all_Zs for single_x # %d \n", i);
			for (int j = 0; j < numLayers-1; j++) {
				Eigen::MatrixXf temp = allZs[i][j];
				std::cout << "\n" << temp.format(CleanFmt) << "\n";
			}
		}
		printf("\n done \n");
	}
	//at this point we have a matrix of z's and matrix of activations.

	//-------------BackWardsPass------------------

	//looks like somewhat repetitive code, but didn't know how else to do it.
	//draw it by hand to check dimensions expected.

	if (debug) { printf("\n BACKWARDS PROPAGATION \n" ); }
	
	//start with populating level 2 by using the error
	Eigen::MatrixXf delta[10];
	for (int i = 0; i < mini_batch_size; i++) {

		if (debug) {
			printf("\n allActivations[%d][2] row = %d   col = %d \n", i, allActivations[i][2].rows(), allActivations[i][2].cols());
			std::cout << mini_Batch_Ys[i].format(CleanFmt);
			printf("\n allZs[%d][1] row = %d   col = %d \n", i, allZs[i][1].rows(), allZs[i][1].cols());
		}

		//SINGLE deltaL2 = (activationL2 - y) times sigmoid zL2 - should have dimensions as BSmatrix 2 by 1
		delta[i] = (costDerivative(allActivations[i][2], mini_Batch_Ys[i])).cwiseProduct(sigmoid_Prime_Vectorial(allZs[i][1]));
	}
	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n delta[%d] : \n", i);
			std::cout << delta[i].format(CleanFmt) << "\n";
		}
	}
	
	for (int i = 0; i < mini_batch_size; i++) { //also down here 2 is the last level
		allGradientsBs[i][1] = delta[i]; //cause 2 is the last level.
		if (debug) { printf("\nallActivations[%d][1].transpose() rows = %d col = %d \n", i, allActivations[i][1].transpose().rows(), allActivations[i][1].transpose().cols()); }
		allGradientsWs[i][1] = delta[i] * allActivations[i][1].transpose(); 
	}

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n allGradientsBs[%d][1] : \n", i);
			std::cout << allGradientsBs[i][1].format(CleanFmt) << "\n";
			printf("\n allGradientsWs[%d][1] : \n", i);
			std::cout << allGradientsWs[i][1].format(CleanFmt) << "\n";
		}
	}
	
	//populate level 1 by using the error as well
	Eigen::MatrixXf deltaL1[10];    

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			//this is supposed to require transposition?!
			printf("\n weightsMatrixL2.transpose() rows = %d cols = %d \n", weightsMatrixL2.transpose().rows(), weightsMatrixL2.transpose().cols());
			std::cout << deltaL1[i].format(CleanFmt) << "\n";
			printf("\n delta[%d] \n", i);
			std::cout << delta[i].format(CleanFmt) << "\n";
			printf("\n allActivations[%d][0] rows = %d cols = %d \n", i, allActivations[i][0].rows(), allActivations[i][0].cols());
			std::cout << allActivations[i][0].format(CleanFmt) << "\n";
		}
	}

	for(int i = 0; i < mini_batch_size; i++){  
	
		deltaL1[i] = (weightsMatrixL2.transpose() * delta[i]).cwiseProduct(allActivations[i][1]); 
	}

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n deltaL1[%d] : \n", i);
			std::cout << deltaL1[i].format(CleanFmt) << "\n";
		}
	}

	for (int i = 0; i < mini_batch_size; i++) { //also down here 2 is the last level
		allGradientsBs[i][0] = deltaL1[i];
		allGradientsWs[i][0] = deltaL1[i] * allActivations[i][0].transpose();
	}

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n allGradientsBs[%d][0] rows = %d cols = %d \n", i, allGradientsBs[i][0].rows(), allGradientsBs[i][0].cols());
			std::cout << allGradientsBs[i][0].format(CleanFmt) << "\n";
			printf("\n allGradientsWs[%d][0] rows = %d cols = %d \n", i, allGradientsWs[i][0].rows(), allGradientsWs[i][0].cols());
			std::cout << allGradientsWs[i][0].format(CleanFmt) << "\n";
		}
	}
}
