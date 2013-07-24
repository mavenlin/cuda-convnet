#include "matrix.h"
#include "nvmatrix.cuh"
#include <iostream>
#include "extrautils.cuh"
#include "thrust/device_vector.h"
#include <iostream>

using namespace std;

int main(int argc, char ** argv)
{

	ifstream fin;
        fin.open(argv[1], std::ios::in);
        if (!fin) exit(1);
        
	float _imgSize, _cases, _channels;
	
	fin >> _imgSize;
	fin >> _cases;
	fin >> _channels;

	int imgSize = (int)_imgSize;
	int cases = (int)_cases;
	int channels = (int)_channels;

	int rows = imgSize * imgSize * channels;
	int cols = cases;

	float * lab = new float[cols];
	float * act = new float[rows * cols];
	float * gra = new float[rows * cols];
	float cost;

	try {
	for(int i=0; i<cols; i++)
		fin >> lab[i];
	for(int i=0; i<cols*rows; i++)
		fin >> act[i];
	for(int i=0; i<cols*rows; i++)
		fin >> gra[i];
	
	fin >> cost;
	}
	catch(std::ifstream::failure e) {
		std::cerr<<"Error";
	}
	fin.close();

	thrust::device_vector<int> counts;
	thrust::device_vector<float> values;
	Matrix labels(lab, 1, cases); // labels as the first input
    Matrix acts(act, rows, cols);   // activations from the previous layer as the second output
    Matrix grad(gra, rows, cols);

    NVMatrix Labels(labels, true);
    NVMatrix Acts(acts, true);
	NVMatrix Grad(grad, true);
    
	values.clear();
	counts.clear();
	sparse_histogram(Labels.getDevData(), cases, values, counts);
	
	NVMatrix sqrts;

	sqrts.resize(channels, values.size());   // Allocate the matrix for summation calculation. The number of rows is equal to the number of channels. 
                                                 // The number of cols is equal to the number of distinct labels.
	sqrts.apply(NVMatrixOps::Zero());

	NVMatrix target;
	target.resize(rows,cols);

	// calculate the cost
	float cost2 = CalculateSqrtSumSquareMatrix(Labels, Acts, values, counts, sqrts, channels, Acts.getNumRows()/channels);
	CalculateGradient(Acts, Labels, sqrts, values, counts, channels, imgSize*imgSize, cases, target);
	target.subtract(Grad);
	target.eltwiseDivide(Grad);
	target.print(rows,cols);
	cout<<cost2<<endl;
	cout<<cost<<endl;


}


