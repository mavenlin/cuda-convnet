# CUDA toolkit installation directory.
export CUDA_INSTALL_PATH=/usr/local/cuda-5.0

# CUDA SDK installation directory.
export CUDA_SDK_PATH=/home/linmin/NVIDIA_CUDA-5.0_Samples

# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=/usr/include/python2.7

# Python library path
export PYTHON_LIB_PATH=/usr/lib

# Python binary path
export PYTHON_BIN_PATH=/usr/bin

# Numpy include directory. This should contain the file arrayobject.h, among others.
export NUMPY_INCLUDE_PATH=/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy

# ATLAS library directory. This should contain the file libcblas.so, among others.
# export ATLAS_LIB_PATH=/usr/lib/atlas-base

export INTEL_MKL_PATH=/opt/intel/mkl

INCLUDE="-I../../../Kernel/include/nvmatrix  -I../../../Kernel/include/matrix  -I../../../PluginsSrc/include/  -I$INTEL_MKL_PATH/include  -I$NUMPY_INCLUDE_PATH  -I$PYTHON_INCLUDE_PATH  -I$CUDA_SDK_PATH/common/inc  -I../../../Kernel/include/layers -I../../../Kernel/include/convnet -I../../../Kernel/include/data -I../../../Kernel/include/conv"
echo $INCLUDE

nvcc -c -g -G -arch sm_20 -o extrautils.cu.o ../../../PluginsSrc/src/extrautils.cu $INCLUDE
#nvcc -c -g -G -arch sm_20 -o nvmatrix.cu.o ../../../Kernel/src/nvmatrix/nvmatrix.cu $INCLUDE
#nvcc -c -g -G -arch sm_20 -o matrix.o ../../../Kernel/src/matrix/matrix.cpp $INCLUDE
#nvcc -c -g -G -arch sm_20 -o nvmatrix_kernels.cu.o ../../../Kernel/src/nvmatrix/nvmatrix_kernels.cu $INCLUDE

nvcc -g -G -arch sm_20 $INCLUDE -o test.bin  test.cu nvmatrix.cu.o nvmatrix_kernels.cu.o matrix.o extrautils.cu.o -L$CUDA_INSTALL_PATH/lib64 -lcublas -L$INTEL_MKL_PATH/lib/intel64 -lmkl_rt

