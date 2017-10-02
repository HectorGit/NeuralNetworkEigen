/*
  ==============================================================================

    Main.cpp
    Created: 29 Sep 2017 3:10:51pm
    Author:  Owner

  ==============================================================================
*/
#include <Eigen/Core>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
//#include "NeuralNetwork.h"

//***where do i get DATA from? from a file? for now I guess it could be a made up matrix in here.

//***wouldn't realistically have TESTING DATA from the start. (The user would make sounds and create 
// the new input vectors).

//***WHEN DOES VALIDATION OCCUR?

//***print the error after each epoch in SGD? 

//***this would need to 'run forever'
//and just keep returning classification
//each time it receives a new instance.

//***should implement a way in which just one new instance
//can be tested with the system (E.G. NON -VECTORIAL)

int main(){

	/*Eigen::Vector3d v(2, 2, 2);
	Eigen::Vector3d w(2, 2, 2);

	double vDotw = v.dot(w); // dot product of two vectors
	Eigen::Vector3d vCrossw = v.cross(w); // cross product of two vectors

	printf("vDotw %d", vDotw);*/

	
	Eigen:: Matrix3f m = Eigen::Matrix3f::Random();
	std::ptrdiff_t i, j;
	float minOfM = m.minCoeff(&i, &j);
	printf("Here is the matrix m:\n");
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	printf("\n %", m.format(CleanFmt));
	printf("Its minimum coefficient is at position (%d,%d)", i, j);

	Eigen::RowVector2f v = Eigen::RowVector2f::Random();
	float maxOfV = v.maxCoeff(&i);
	printf("\nHere is the vector v:\n");
	//printf(v);
	printf("Its maximum coefficient is at position (%d)", i);
	
	
	/*SGD(training_data, epochs, mini_batch_size, eta,test_data)*/
	//myNetwork.SGD(trainData, 30, 10, 3.0, testData); //30 epochs, //10 vectors for mini batch, 3.0 error, 
	return 0;
}






