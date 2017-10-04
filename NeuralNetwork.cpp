/*
  ==============================================================================

    NeuralNetwork.cpp
    Created: 29 Sep 2017 2:54:03pm
    Author:  Owner

	//VERSION 1 OF CODE BASED ON ALEX THOMO'S IMPLEMENTATION

  ==============================================================================
*/

#include <Eigen/Core>
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
	bool debug = false;
	//printf("what is going on?");
	numLayers = paramSizes.size();
	if (debug) { printf("\n numLayers is: %d \n", numLayers); }

	eta = 0; 
	sizes = paramSizes; //NOT SURE IF THIS WORKS //MIGHT NEED TO COPY IT IN SOMEHOW.

	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) { printf("\n------------Initializing NeuralNet------------\n"); }

	if (debug) { printf("\nLEVEL ONE\n"); }

	weightsMatrixL1 = Eigen::MatrixXf::Random(sizes[0], sizes[1]);
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

	weightsMatrixL2 = Eigen::MatrixXf::Random(sizes[1], sizes[2]);
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

void NeuralNetwork::feedForward(Eigen::MatrixXf &x/*Eigen::MatrixXf &activationL1, Eigen::MatrixXf &activationL2*/)
{
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) {
		printf("--------feedforward algorithm------");
		printf("\n----------LEVEL ONE ---------\n");
	}

	Eigen::MatrixXf dotProductL1 = weightsMatrixL1.transpose()*x;
	if (debug) {
		printf("\n wT dot x \n");
		std::cout << "\n wT: \n" << weightsMatrixL1.transpose().format(CleanFmt);
		std::cout << "\n x: \n" << x.format(CleanFmt);
		std::cout << "\n Result: \n" << dotProductL1.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	Eigen::MatrixXf sumL1 = dotProductL1 + biasesMatrixL1;
	if (debug) {
		printf("\n + Biases \n");
		std::cout << "\n b: \n" << biasesMatrixL1.format(CleanFmt);
		std::cout << "\n Result: \n" << sumL1.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	activationL1 = sigmoid_Vectorial(sumL1);
	if (debug) {
		printf("\n\n Apply Sigmoid\n");
		std::cout << "\n Result: \n" << activationL1.format(CleanFmt);
	}

	if (debug) { printf("\n\n---------LEVEL TWO ----------\n"); }

	Eigen::MatrixXf dotProductL2 = weightsMatrixL2.transpose()*activationL1;
	if (debug) {
		printf("\n wT dot activationL1 \n");
		std::cout << "\n wT: \n" << weightsMatrixL2.transpose().format(CleanFmt);
		std::cout << "\n activationL1: \n" << activationL1.format(CleanFmt);
		std::cout << "\n Result: \n" << dotProductL2.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	Eigen::MatrixXf sumL2 = dotProductL2 + biasesMatrixL2;
	if (debug) {
		printf("\n + Biases \n");
		std::cout << "\n b: \n" << biasesMatrixL2.format(CleanFmt);
		std::cout << "\n Result: \n" << sumL2.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	activationL2 = sigmoid_Vectorial(sumL2);
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
		//delta_gradient_B = backPropagatio(x,y)

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

//consideration - pass in the gradient matrices? // BY REFERERNCE !!!! They are members already. - consider removing them as params later
/*void NeuralNetwork::backPropagation(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2)
{
	backPropagationForWs( x,  y,  gradient_Ws, gradient_Bs);
	backPropagationForBs( x, y,  gradient_Ws, gradient_Bs);
}*/

/*void NeuralNetwork::backPropagationForWs(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2)
{
	//create the gradient Matrices
	//gradient_W, gradient_B = zeroes matrices;

	//-------------FeedForward------------------
	//activation defined in private members? NO !
	//activation = x; //both vectors - init activation here.

	//activations = [x]; //start to put things into this vector

	//zs = []; // store all the z vectors layer by layer
			 // one z = wx+b in vector form (all z's for one level)

			 // for each bias and weight
			 //calculate z = w dot activation + b (all are vectors)
			 //zs.append z;
			 //calculate the activation_ by sigmoid(z)
			 //activations append activation

			 //->>>end up w - matrix of z's and matrix of activations.

			 //-------------BackWardsPass------------------

			 //START BY get the ERROR (delta) at the last level ([-1]) 
			 //indexing DOES NOT WORKLIKE THAT ON C++?!!!

			 // and compute the other errors (partial drvs of. W's and B's)

			 // error/delta = cost_derivative(activations[-1]) * sigmoid_prime(zs[-1]); //y-x

			 //HERE WE SEE WE MIGHT SEPARATE BACKPROP INTO A) for B's ---and--- B)  for W's
			 // gradient_B[-1] = delta;
			 // gradient_W[-1] = delta dot activations[-2].transpose //previous level activation; 

			 //-------Rest of the levels--------//
			 //REVIEW THIS MATH !!
			 // for each level after that -
			 // get its z -> z = z[-L]
			 // and get its sp = sigmoid_prime(z)
			 // get delta = ( weightsMatrix[-L + 1].transpose dot delta ) * SP;
			 // gradient_B[-L] = delta;
			 // gradient_W[-L] = (delta dot activations[-L-1].transpose);

			 //return gradient_B and gradient_W; 
			 //-> TIP == separate into TWO METHODS (one for B, 
			 // one for W's)
}*/

/*void NeuralNetwork::backPropagationForBs(Eigen::MatrixXf x, Eigen::MatrixXf y, Eigen::MatrixXf &gradientWsL1, Eigen::MatrixXf &gradientBsL1, Eigen::MatrixXf &gradientWsL2, Eigen::MatrixXf &gradientBsL2)
{
			
}*/

