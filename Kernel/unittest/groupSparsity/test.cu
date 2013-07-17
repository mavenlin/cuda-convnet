#include "matrix.h"
#include "nvmatrix.cuh"
#include <iostream>
#include "extrautils.cuh"
#include "thrust/device_vector.h"

using namespace std;

int main(int argc, char ** argv)
{
	float ten[10] = {0,1,2,2,1,1,0,1,0,2};
	float act[40];
	for (int i=0; i<40; i++)
		act[i] = i+1;

	int _channels = 2;

	thrust::device_vector<int> counts;
	thrust::device_vector<float> values;
	Matrix labels(ten, 1, 10); // labels as the first input
    Matrix acts(act, 4, 10);   // activations from the previous layer as the second output
    
    NVMatrix Labels(labels, true);
    NVMatrix Acts(acts, true);

    int numCases = Labels.getNumElements();
    
	values.clear();
	counts.clear();
	sparse_histogram(Labels.getDevData(), numCases, values, counts);
	
	NVMatrix sqrts;

	sqrts.resize(_channels, values.size());   // Allocate the matrix for summation calculation. The number of rows is equal to the number of channels. 
                                                 // The number of cols is equal to the number of distinct labels.
	sqrts.apply(NVMatrixOps::Zero());

	NVMatrix target;
	target.resize(4,10);

	// calculate the cost
	float cost = CalculateSqrtSumSquareMatrix(Labels, Acts, values, counts, sqrts, _channels, Acts.getNumRows()/_channels);

	CalculateGradient(Acts, Labels, sqrts, values, counts, _channels, 2, 10, target);

	target.print(10,10);
	cout<<cost<<endl;

}

