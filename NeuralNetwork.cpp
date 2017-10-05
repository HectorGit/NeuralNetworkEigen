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

//Constructor
NeuralNetwork::NeuralNetwork(std::vector<int> &paramSizes)
{
	bool debug = true;
	//printf("what is going on?");
	numLayers = paramSizes.size();
	if (debug) { printf("\n numLayers is: %d \n", numLayers); }

	eta = 0; 
	sizes = paramSizes; //NOT SURE IF THIS WORKS //MIGHT NEED TO COPY IT IN SOMEHOW.

	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) { printf("\n------------Initializing NeuralNet------------\n"); }

	if (debug) { printf("\nLEVEL ONE\n"); }

	weightsMatrixL1 = Eigen::MatrixXf::Random(sizes[1], sizes[0]); //check this
	if (debug) {printf("\nHere's weightsMatrixL1\n");
		std::cout << weightsMatrixL1.format(CleanFmt);}
	gradientWsL1 = Eigen::MatrixXf::Zero(sizes[0], sizes[1]);
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
	gradientWsL2 = Eigen::MatrixXf::Zero(sizes[1], sizes[2]);
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
	bool debug = true;
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

/*void NeuralNetwork::stochasticGradientDescent(Eigen::MatrixXf &trainingData, int epochs, int miniBatchSize, float eta, Eigen::MatrixXf &testData)
{
	//if(testData){ n = len(testData;}

	//for every epoch
		// shuffle the training Data
			// alternative to random.shuffle
			// is to have a vector with numbers that
			// will contain randomized indexes. this vector
			//can be used to access the training Data randomly
			//even if it it un shuffled.
		//select mini-batches
			//mini_batches = [
			//training_data[k:k+mini_batch_size]
			//for k in xrange(0, n, mini_batch_size)]
		// for each batch 
			//updateNetwork(mini_batch, eta) //backpropagation
}*/

//***Consideration- need to have access to NN :: biasesMatrix and weightMatrix
/*void NeuralNetwork::updateMiniBatch(Eigen::MatrixXf &mini_batch, float eta, Eigen::MatrixXf &biasesMatrixL1, Eigen::MatrixXf &weightsMatrixL1, Eigen::MatrixXf &biasesMatrixL2, Eigen::MatrixXf &weightsMatrixL2)
{
	//create the gradient_W and gradient B matrices (fill w Zeros)
	//for X, Y in mini batch
		//unclear if could get both from a single
		//backPropagation operation !!!
		//delta_gradient_W = backPropagation(x,y)
		//delta_gradient_B = backPropagation(x,y)

		    //delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            //nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            //nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		
		//elementwise
		//gradient_W = gradient_W + delta_gradient_W;
		//gradient_B = gradient_B + delta_gradient_B;
		//This changes the gradient matrices from being zero into having values 

	//self.weights = [w - (eta / len(mini_batch))*nw
	//for w, nw in zip(self.weights, nabla_w)]
	//self.biases = [b - (eta / len(mini_batch))*nb
	//for b, nb in zip(self.biases, nabla_b)]
		//This applies the gradients the corresponding matrices.

		//elementwise
		// weightsMatrices = weightsMatrices - gradientW;
		// biasesMatrices = biasesMatrices - gradientB;
}*/

//pass in y?
void NeuralNetwork::backPropagation(int mini_batch_size)
{	
	bool debug = true;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	//vectors didn't want to work... :( only 1-D
	//should these be class members in NeuralNetwork.h?
	//std::vector< std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > all_gradientsBs;
	Eigen::MatrixXf allGradientsBs[10][2]; //nabla_b
	//std::vector< std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > all_gradientsWs;
	Eigen::MatrixXf allGradientsWs[10][2]; //nabla_w
	//std::vector< std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > all_activations;
	Eigen::MatrixXf allActivations[10][3];
	//std::vector< std::vector< Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf> > all_Zs;
	Eigen::MatrixXf allZs[10][2];
	//Eigen::MatrixXf allWeightsMatrices[10][2];

	//STEP 1 - 
	for (int i = 0; i < mini_batch_size; i++) {
		if (debug){ printf("\n feedforward for all_Xs[%d] \n", i); }
		feedForward(all_Xs[i], all_Ys[i]); //does this work?
		
		allActivations[i][0] = all_Xs[i];
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
			std::cout << all_Ys[i].format(CleanFmt);
			printf("\n allZs[%d][1] row = %d   col = %d \n", i, allZs[i][1].rows(), allZs[i][1].cols());
		}

		//SINGLE deltaL2 = (activationL2 - y) times sigmoid zL2 - should have dimensions as BSmatrix 2 by 1
		delta[i] = (costDerivative(allActivations[i][2], all_Ys[i])).cwiseProduct(sigmoid_Prime_Vectorial(allZs[i][1]));
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
