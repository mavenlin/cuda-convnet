#ifndef EXTRA_LAYERS
#define EXTRA_LAYERS

#include "layer.cuh"
#include "neuron.cuh"
#include "extrautils.cuh"
#include "nvmatrix_operators.cuh"

/***************************************
 * Group Sparsity Cost in Sample Domain for each feature map
 ***************************************/

class GroupSparsityInLabelCostLayer : public CostLayer {
protected:
	int _channels, _imgSize;
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

    virtual void _activate(PASS_TYPE passType) {
    	
        if (passType == PASS_TRAIN) {
            // If the _dropout is not the same size as the input, resize the dropout to match the input
            if(!_inputs->isSameDims(*_dropout))
                _dropout->resize(*_inputs);

            // randomize the dropouts in between [0-1] with uniform distribution
            // TODO Check whether we need to initialize anything to make the random number generater work
            _dropout->randomizeUniform();

            // set the elements at the position where _dropout[position] < _dropoutprob
            // ? How to share the output from the inputs?
            _inputs->applyBinary(DropoutOperator(_dropoutprob), *_dropout, *_outputs);
        }
        else { // eithter PASS_GC or PASS_TEST, because for PASS_GC, the network structure is not supposed to change in fprop
            _inputs->apply(NVMatrixOps::MultByScalar(1 - _dropoutprob), *_outputs);
        }
    	
    }

    virtual void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target, PASS_TYPE passType) {

        if (passType == PASS_TRAIN)
    	   // Also set part of the gradients to zero at the same position as the activations
            actsGrad.applyBinary(DropoutOperator(_dropoutprob), *_dropout, target);
        else
            // This condition will never be called by the PASS_TEST
            // it is here in case of PASS_CG
            actsGrad.apply(NVMatrixOps::MultByScalar(1 - _dropoutprob), target);
    }
    
    virtual void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target, PASS_TYPE passType) {
        if (passType == PASS_TRAIN)
            actsGrad.applyTernary(AddGradientBinaryOperator<DropoutOperator>(DropoutOperator(_dropoutprob)), *_dropout, target, target);
        else
            // This condition will never be called by the PASS_TEST
            // it is here in case of PASS_CG
            actsGrad.applyBinary(AddGradientOperator<NVMatrixOps::MultByScalar>(NVMatrixOps::MultByScalar(1 - _dropoutprob)), target, target);
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
		
        __device__ inline float operator()(float source, float uniform) const {
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


/************************************
 * KL Divergence neuron Use together with SumCostLayer for KL divergence penalty
 ************************************/
class KLNeuron : public Neuron {
protected:
    float _p;
    virtual void _activate(PASS_TYPE passType) { // act = w*log(w/p)-w+p
        
    }

    virtual void _computeInputGrad(NVMatrix& actsGrad, NVMatrix& target, PASS_TYPE passType) {

    }
    
    virtual void _addInputGrad(NVMatrix& actsGrad, NVMatrix& target, PASS_TYPE passType) {
        
    }
};

/************************************
 * L1 neuron
 ************************************/



/************************************
 * Sum Cost
 ************************************/
class SumCostLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
        _costv.clear();
        _costv.push_back(_inputs[0]->sum());
    }
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
        _prev[inpIdx]->getActsGrad().scale(scaleTargets);
        _prev[inpIdx]->getActsGrad().addScalar(-_coeff*1); // The gradient is always 1
    }
public:
    SumCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
    }
};

/****************************************
 * Weight abs sum
 ****************************************/
// This layer does not use the activation of the previous layer or the activation of its own.
// This layer does not use the activation of the previous layer, which means, this layer does not use the input. Because, indeed this layer uses the previous layer's weight.
// Be careful of the addition to the previous layer's weights, because the weights are first cached and when update weights are called, it is update, thus should not add directly.
// Also double check whether the weights are already changed in in between fprop and bprop of this layer. 
class WeightAbsSumLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
        assert(_prev.size() == 1); // The size of the _prev is not 1, that means there are more than one inputs to this layer, Currently it is not supported.
        WeightLayer* prev_weight = (WeightLayer*) _prev[0]; 
        NVMatrix& weight_matrix = prev_weight->getWeights(0).getW();
        weight_matrix.apply(NVMatrixOps::Abs(), getActs()); // take the abs of the weights in the previous layer, and then the next step is to sum.
        _costv.clear();
        _costv.push_back(getActs().sum());
    }
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
        WeightLayer* prev_weight = (WeightLayer*) _prev[0];
        NVMatrix& weight_grad = prev_weight->getWeights(0).getGrad();
        NVMatrix& weight_matrix = prev_weight->getWeights(0).getW();
        // weight_matrix.eltwiseDivide(weight_grad.apply(), weight_grad);
        // weight_grad.add();
        // TODO: not finished yet.
    }
public:
    WeightAbsSumLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {
    }
};


#endif
