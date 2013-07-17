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

// numBlocksX = DIVUP(numCases, B_X)
// numBlocksY = channels * DIVUP(imagePixels, B_Y*PixelsPerThread)
// temporary (numBlocksY, numBlocksX*numLabels)
// Pass the num of Labels as the init size of the hist shared memory
template <int B_Y, int B_X, int PixelsPerThread>
__global__ void CalculateSqrtSumSquare(float * labels, float * acts, float * values, int * counts, float * target, float * temp, int channels, int imagePixels, int numCases, int numLabels)
{
	extern __shared__ float hist[];
	__shared__ float shLabel[B_X];
	__shared__ float partial[B_Y][B_X];
	partial[threadIdx.y][threadIdx.x] = 0;
	if (threadIdx.y == 0)
		shLabel[threadIdx.x] = labels[threadIdx.x + blockIdx.x * B_X];
	__syncthreads();

	int numBlocksPerChannel = DIVUP(imagePixels, B_Y * PixelsPerThread);
	int blkIdxInChannel = blockIdx.y % numBlocksPerChannel;
	int channelIndex = blockIdx.y / numBlocksPerChannel;
	int numBlocksX = DIVUP(numCases, B_X);

	acts += threadIdx.x + blockIdx.x * B_X \             // offset by cases
			threadIdx.y * numCases + \                   // offset by thread in one block
			channelIdx * imagePixels * numCases + \      // offset by channel
			blkIdxInChannel * B_Y * PixelsPerThread * numCases;     //offset by block inside of channel

	#pragma unroll
	for (int i=0; i<imagePixels; i+=PixelsPerThread) {
		if (blkIdxInChannel * B_Y * PixelsPerThread + threadIdx.y + i < imagePixels)
			partial[threadIdx.y][threadIdx.x] += acts[i*numCases] * acts[i*numCases];
	}
	__syncthreads();

	// Since now the data are all in the shared memory, we don't need to consider the coalesced operation on the memory.
	int tidx = threadIdx.y * B_X + threadIdx.x;
	int numRounds = DIVUP(numLabels, B_X * B_Y);
	for (int i=0; i<numLabels; i+=B_X*B_Y)
		if (i+tidx<numLabels)
			hist[i+tidx] = 0;

	for (int i=0; i<numLabels; i+=B_X*B_Y) {
		#pragma unroll
		for (int j=0; j<B_X; j++) {
			if (i + tidx < numLabels && shLabel[j] == values[i+tidx])
				#pragma unroll
				for (int k=0; k<B_Y; k++)
					hist[i+tidx] += partial[k][j];
		}
		hist[i+tidx] *= counts[i+tidx];
	}
	__syncthreads();

	float * tmp = temp + (channelIdx * numBlocksPerChannel * numBlocksX + blkIdxInChannel * numBlocksX + blockIdx.x) * numLabels;
	for (int i=0; i<numLabels; i+=B_X*B_Y)
		if (tidx+i<numLabels)
			tmp[tidx+i] = hist[tidx+i];
	__syncthreads();

	int chan = blockIdx.y;
	int labelIdx = threadIdx.x + B_X * blockIdx.x;
	if (chan < channels){
		if (threadIdx.y == 0 && labelIdx < numLabels) {
			target[chan*numLabels+labelIdx] = 0;
			for (int i=0; i<numBlocksX*numBlocksPerChannel; i++)
				target[chan*numLabels+labelIdx] += temp[(chan*numBlocksX*numBlocksPerChannel+i)*numLabels+labelIdx];
			target[chan*numLabels+labelIdx] = sqrt(target[chan*numLabels+labelIdx]);
		}
	}

}

float CalculateSqrtSumSquareMatrix(NVMatrix& labels, NVMatrix& acts, thrust::device_vector<float>& values, thrust::device_vector<int>& counts, NVMatrix& target, int channels, int imagePixels)
{
	assert(acts.getNumRows() == channels*imagePixels);

	int B_X = 32;
	int B_Y = 8;
	int numCases = acts.getNumCols();
	int PixelsPerThread = 32;
	int gridydim = channels * DIVUP(imagePixels, B_Y*PixelsPerThread);
	int gridxdim = DIVUP(numCases, B_X);

	int numBlocksPerChannel = DIVUP(imagePixels, B_Y * PixelsPerThread);
	int numBlocksX = DIVUP(numCases, B_X);
	
	NVMatrix temp(numBlocksX*numBlocksPerChannel*channels, numLabels);

	dim3 blocks(gridxdim, gridydim); // The dimension of the grid
	dim3 threads(B_X, B_Y);            // The dimension of the block, 32 x 16 = 512, which is the thread number available inside a block for compute compatibility<2.0.
	float * values_ptr = thrust::raw_pointer_cast(values.data());
	int * counts_ptr = thrust::raw_pointer_cast(counts.data());
	CalculateSqrtSumSquare<8, 32, 32><<<blocks, threads, counts.size()>>>(labels.getDevData(), acts.getDevData(), values_ptr, counts_ptr, target.getDevData(), temp.getDevData(), channels, imagePixels, numCases, counts.size());
	return target.sum();
}


// Blocks Y are used for pixels and channels
// numBlocksY = channels * DIVUP(imgPixels, PixelsPerThread * B_Y)
// Blocks X are used for Samples
// numBlocksX = DIVUP(numCases, B_X)
// The shared memory size should be three times the size of numLabels.
template <int B_Y, int B_X, int PixelsPerThread>
__global__ void kCalculateGradient(float * acts, float * labels, float * sqrts, float * values, int * counts, int numLabels, int channels, int imagePixels, int numCases, float * target)
{
	int numBlocksPerChannel = DIVUP(imagePixels, B_Y * PixelsPerThread);
	int blkIdxInChannel = blockIdx.y % numBlocksPerChannel;
	int channelIndex = blockIdx.y / numBlocksPerChannel;
	int numBlocksX = DIVUP(numCases, B_X);

	extern __shared__ float shSqrt[];

	int tidx = threadIdx.x + threadIdx.y * B_X;
	for (int i=0; i<numLabels; i+=B_X*B_Y)
		if (i + tidx < numLabels) {
			shSqrt[i+tidx] = sqrts[numLabels*channelIndex + i + tidx];
			shSqrt[i+tidx+numLabels] = values[i+tidx];
			shSqrt[i+tidx+2*numLabels] = counts[i+tidx];
		}
	__syncthreads();

	target += (channelIndex * imagePixels + blkIdxInChannel * B_Y * PixelsPerThread  + threadIdx.y) * numCases + threadIdx.x + B_X * blockIdx.x;
	acts += (channelIndex * imagePixels + blkIdxInChannel * B_Y * PixelsPerThread + threadIdx.y) * numCases + threadIdx.x + B_X * blockIdx.x;

	int caseIndex = threadIdx.x + B_X * blockIdx.x;
	if (caseIndex < numCases) {
		for (int i=0; i<imagePixels; i+=PixelsPerThread)
			if (blkIdxInChannel * B_Y * PixelsPerThread + threadIdx.y + i < imagePixels)
				for (int j=0; j<numLabels; j++){
					if (shSqrt[numLabels+j]==labels[caseIndex])
						target[i*numCases] = shSqrt[2*numLabels+j]*acts[i*numCases] / (shSqrt[j] + 2.5e-7); // EPS
				}
	}

}

void CalculateGradient(NVMatrix& acts, NVMatrix& labels, NVMatrix& sqrts, thrust::device_vector<float>& values, thrust::device_vector<int>& counts, int channels, int imagePixels, int numCases, NVMatrix& target)
{
	int B_X = 32;
	int B_Y = 8;
	int numCases = acts.getNumCols();
	int PixelsPerThread = 16;
	int gridydim = channels * DIVUP(imagePixels, B_Y*PixelsPerThread);
	int gridxdim = DIVUP(numCases, B_X);
	dim3 blocks(gridxdim, gridydim); // The dimension of the grid
	dim3 threads(B_X, B_Y);            // The dimension of the block, 32 x 16 = 512, which is the thread number available inside a block for compute compatibility<2.0.
	kCalculateGradient<8,32,16><<<blocks, threads, counts.size()>>>(acts.getDevData(), labels.getDevData(), sqrts.getDevData(), thrust::raw_pointer_cast(values.data()), thrust::raw_pointer_cast(counts.data()), counts.size(), channels, imagePixels, numCases, target.getDevData());
}
