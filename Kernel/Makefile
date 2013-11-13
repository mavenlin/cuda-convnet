MODELNAME := libkernel

INCLUDES :=  -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) -I./include/data -I./include/layers -I./include/matrix -I./include/conv -I./include/nvmatrix -I./include/convnet -I$(INTEL_MKL_PATH)/include -I$(CUDA_SDK_PATH)/common/inc
LIB := -lpthread -L$(CUDA_INSTALL_PATH)/lib64 -L$(INTEL_MKL_PATH)/lib/intel64 -lmkl_rt -ldl

USECUBLAS   := 1
USE_MKL     := 1

PYTHON_VERSION=$(shell $(PYTHON_BIN_PATH)/python -V 2>&1 | cut -d ' ' -f 2 | cut -d '.' -f 1,2)
LIB +=  -L$(PYTHON_LIB_PATH) -lpython$(PYTHON_VERSION)

GENCODE_ARCH := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
COMMONFLAGS := -DNUMPY_INTERFACE -DMODELNAME=$(MODELNAME) -DINITNAME=init$(MODELNAME)

EXECUTABLE	:= $(MODELNAME).so

CUFILES				:= $(shell echo src/data/*.cu src/conv/*.cu src/nvmatrix/*.cu src/layers/*.cu src/convnet/*.cu)
CU_DEPS				:= $(shell echo include/data/*.cuh include/conv/*.cuh include/nvmatrix/*.cuh include/layers/*.cuh include/convnet/*.cuh)
CCFILES				:= $(shell echo src/matrix/*.cpp)
C_DEPS				:= $(shell echo include/matrix/*.h include/convnet/*.h)

include common-gcc-cuda-4.0.mk
	
makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)/src/conv
	$(VERBOSE)mkdir -p $(OBJDIR)/src/nvmatrix
	$(VERBOSE)mkdir -p $(OBJDIR)/src/data
	$(VERBOSE)mkdir -p $(OBJDIR)/src/matrix
	$(VERBOSE)mkdir -p $(OBJDIR)/src/layers
	$(VERBOSE)mkdir -p $(OBJDIR)/src/convnet	
	$(VERBOSE)mkdir -p $(TARGETDIR)
