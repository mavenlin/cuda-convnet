#ifndef EXTRA_LAYERS
#define EXTRA_LAYERS

#include "layer.cuh"
#include "neuron.cuh"
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


/**************************************
 * Random Dropout Neuron
 **************************************/
// In the corresponding python code, 
// remember to set the property useActsGrad and useInput to 0 as this dropoutlayer uses none of the two
// use this neuron together with the neuron layer to create a dropout layer


class DropoutNeuron : public Neuron {
protected:
	NVMatrix * _dropout;
	float _dropoutprob; // This should be initialized in the Neuron Constructor

    void _activate() {
    	
    	// If the _dropout is not the same size as the input, resize the dropout to match the input
    	if(!_inputs->isSameDims(*_dropout))
    		_dropout->resize(*_inputs);

    	// randomize the dropouts in between [0-1] with uniform distribution
    	_dropout->randomizeUniform();

    	// set the elements at the position where _dropout[position] < _dropoutprob
        _inputs->applyBinary(DropoutOperator(_dropoutprob), *_dropout, *_outputs);
    }

    void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target) {

    	// Also set part of the gradients to zero at the same position as the activations
        actsGrad.applyBinary(DropoutOperator(_dropoutprob), *_dropout, target);
    }
    
    void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target) {
        actsGrad.applyTernary(AddGradientBinaryOperator<DropoutOperator>(DropoutOperator(_dropoutprob)), *_dropout, target, target);
    }

public:
	class DropoutOperator {
	private:
		float _prob;
	public:
		DropoutOperator(float prob) : _prob(prob) {
			assert(prob>0);
			assert(prob<1);
		}
		__device__ inline float operator()(float source, float uniform) {
			if( uniform < _prob )
				return 0;
			else
				return source;
		}
	};
    
    DropoutNeuron(PyObject* neuronParamsDict) : Neuron() {
    	_dropout = new NVMatrix();
    	_dropoutprob = pyDictGetFloat(neuronParamsDict, "prob");
    }
};




#endif