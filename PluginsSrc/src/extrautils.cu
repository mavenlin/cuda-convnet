#include "extrautils.cuh"


void sparse_histogram(float * input, int num, device_vector<float>& histogram_values, device_vector<int>& histogram_counts)
{
	typedef float ValueType; // input value type
	typedef int IndexType;   // histogram index type

	// copy input data
	thrust::device_ptr<float> d_ptr(input);
	thrust::device_vector<ValueType> data(d_ptr, d_ptr + num);

	// print the initial data
	// print_vector("initial data", data);

	// sort data to bring equal elements together
	thrust::sort(data.begin(), data.end());

	// print the sorted data
	// print_vector("sorted data", data);

	// number of histogram bins is equal to number of unique values (assumes data.size() > 0)
	IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
		data.begin() + 1,
		IndexType(1),
		thrust::plus<IndexType>(),
		thrust::not_equal_to<ValueType>());

	// resize histogram storage
	histogram_values.resize(num_bins);
	histogram_counts.resize(num_bins);

	// compact find the end of each bin of values
	thrust::reduce_by_key(data.begin(), data.end(), thrust::constant_iterator<IndexType>(1), histogram_values.begin(), histogram_counts.begin());

	// print the sparse histogram
	// print_vector("histogram values", histogram_values);
	// print_vector("histogram counts", histogram_counts);
}


// target is of size channels x numLabels
// counts and values is of size 1 x numLabels
// acts is of size channels*imagePixels x numCases
// labels is of size 1 x numCases
// B_Y and B_X is the dimension of the block.
template <int B_Y, int B_X>
__global__ void CalculateSqrtSumSquare(float * labels, float * acts, float * values, int * counts, float * target, int channels, int imagePixels, int numCases, int numLabels)
{
	int labelIdx      = blockIdx.x * B_X + threadIdx.x;
	if(labelIdx >= numLabels)
		return;
	float label       = values[labelIdx];
	int channelIdx    = blockIdx.y * B_Y + threadIdx.y;
	if(channelIdx >= channels)
		return;
	int pixelStartIdx = channelIdx * imagePixels;
	int targetIdx     = numLabels * channelIdx + labelIdx;
	for(int i = 0; i < numCases; i++){
		if(labels[i] == label)
			for(int j = 0; j < imagePixels; j++){
				target[targetIdx] += pow(acts[(pixelStartIdx+j)*numCases + i], 2);
			}
	}
	// multiply by the length of the group
	target[targetIdx] *= counts[labelIdx];
	target[targetIdx] = sqrt(target[targetIdx]);
}



float CalculateSqrtSumSquareMatrix(NVMatrix& labels, NVMatrix& acts, thrust::device_vector<float>& values, thrust::device_vector<int>& counts, NVMatrix& target, int channels, int imagePixels)
{
	assert(acts.getNumRows() == channels*imagePixels);

	int gridydim = DIVUP(channels, 16);
	int gridxdim = DIVUP(counts.size(), 32);
	dim3 blocks(gridxdim, gridydim); // The dimension of the grid
	dim3 threads(32, 16);            // The dimension of the block, 32 x 16 = 512, which is the thread number available inside a block for compute compatibility<2.0.
	float * values_ptr = thrust::raw_pointer_cast(values.data());
	int * counts_ptr = thrust::raw_pointer_cast(counts.data());
	CalculateSqrtSumSquare<16,32><<<blocks, threads>>>(labels.getDevData(), acts.getDevData(), values_ptr, counts_ptr, target.getDevData(), channels, imagePixels, labels.getNumElements(), counts.size());
	return target.sum();
}

template <int B_Y, int B_X>
__global__ void kCalculateGradient(float * acts, float * labels, float * sqrts, float * values, int * counts, int numLabels, int channels, int imagePixels, int numCases, float * target)
{
	int labelIdx = blockIdx.x * B_X + threadIdx.x;
	if(labelIdx >= numLabels)
		return;
	float label = values[labelIdx];
	int channelIdx = blockIdx.y * B_Y + threadIdx.y;
	if(channelIdx >= channels)
		return;
	int pixelStartIdx = channelIdx * imagePixels;
	for(int i = 0; i < numCases; i++){
		if(labels[i] == label)
			for(int j = 0; j < imagePixels; j++){
				int targetIdx = (pixelStartIdx+j)*numCases + i;
				target[targetIdx] = acts[targetIdx]/sqrts[channelIdx*numLabels+labelIdx]; // TODO: handle the case when the denominator is very small.
			}
	}
}

void CalculateGradient(NVMatrix& acts, NVMatrix& labels, NVMatrix& sqrts, thrust::device_vector<float>& values, thrust::device_vector<int>& counts, int channels, int imagePixels, int numCases, NVMatrix& target)
{
	int gridydim = DIVUP(channels, 16);
	int gridxdim = DIVUP(counts.size(), 32);
	dim3 blocks(gridxdim, gridydim); // The dimension of the grid
	dim3 threads(32, 16);            // The dimension of the block, 32 x 16 = 512, which is the thread number available inside a block for compute compatibility<2.0.
	kCalculateGradient<16,32><<<blocks, threads>>>(acts.getDevData(), labels.getDevData(), sqrts.getDevData(), thrust::raw_pointer_cast(values.data()), thrust::raw_pointer_cast(counts.data()), counts.size(), channels, imagePixels, numCases, target.getDevData());
}
