#ifndef EXTRAUTILS
#define EXTRAUTILS

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

#include "nvmatrix.cuh"

using namespace thrust;

template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
	typedef typename Vector::value_type T;
	std::cout << "  " << std::setw(20) << name << "  ";
	thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
	std::cout << std::endl;
}


void sparse_histogram(float * input, int num, device_vector<float>& histogram_values, device_vector<int>& histogram_counts);

float CalculateSqrtSumSquareMatrix(NVMatrix& labels, NVMatrix& acts, thrust::device_vector<float>& values, thrust::device_vector<int>& counts, NVMatrix& target, int channels, int imagePixels);

void CalculateGradient(NVMatrix& acts, NVMatrix& labels, NVMatrix& sqrts, thrust::device_vector<float>& values, thrust::device_vector<int>& counts, int channels, int imagePixels, int numCases, NVMatrix& target);

#endif