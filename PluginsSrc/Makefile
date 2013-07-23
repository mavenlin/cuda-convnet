MODELNAME := extralayers

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I../Kernel/include/data -I../Kernel/include/layers -I../Kernel/include/matrix -I../Kernel/include/conv -I../Kernel/include/nvmatrix -I../Kernel/include/convnet -I./include -I$(INTEL_MKL_PATH)/include -I$(CUDA_SDK_PATH)/common/inc
LIB := -lpthread -L$(CUDA_INSTALL_PATH)/lib64 -L$(INTEL_MKL_PATH)/lib/intel64 -lmkl_rt -ldl

USECUBLAS   := 1
USE_MKL     := 1

PYTHON_VERSION=$(shell $(PYTHON_BIN_PATH)/python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB +=  -L$(PYTHON_LIB_PATH) -lpython$(PYTHON_VERSION) -L../Kernel/bin/linux/release -lkernel

GENCODE_ARCH := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/*.cu)
CU_DEPS				:= $(shell echo include/*.cuh)

include common-gcc-cuda-4.0.mk
	
makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src
	$(VERBOSE)mkdir -p $(TARGETDIR)
