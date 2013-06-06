#ifndef EXTRA_LAYERS
#define EXTRA_LAYERS

#include "layer.cuh"
#include "extrautils.cuh"

/***************************************
 * Group Sparsity Cost in Sample Domain for each feature map
 ***************************************/

class GroupSparsityInLabelCostLayer : public CostLayer {
protected:
	intv *_channels, *_imgSize;
	NVMatrix sqrts;
	thrust::device_vector<int> counts;
	thrust::device_vector<float> values;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    GroupSparsityInLabelCostLayer(ConvNet* convNet, PyObject* paramsDict);
};


#endif