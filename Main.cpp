/*
  ==============================================================================

    Main.cpp
    Created: 29 Sep 2017 3:10:51pm
    Author:  Owner

  ==============================================================================
*/
#include <Eigen/Core>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <iostream>

#include "NeuralNetwork.h"

int main(){

	/*sample code - instantiate a vector of random numbers and print it
	Eigen::RowVectorXf v = Eigen::RowVectorXf::Random(1,2);//1st argument always 1
	std::cout << v.format(CleanFmt);*/

	std::vector<int> paramSizes = { 2,3,2 };
	NeuralNetwork* aNet = new NeuralNetwork(paramSizes);
	
	/*SGD(training_data, epochs, mini_batch_size, eta,test_data)*/
	//myNetwork.SGD(trainData, 30, 10, 3.0, testData); //30 epochs, //10 vectors for mini batch, 3.0 error, 
	return 0;
}






