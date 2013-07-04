#include "extralayers.cuh"
#include "plugin.cuh"
#include <iostream>

/***************************************
 * Group Sparsity Cost in Sample Domain
 ***************************************/

// protected
void GroupSparsityInLabelCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType){
	// there is no need to worry about sharing. 
	// They are already handled in Layer's fprop function and the python script will judge whether the input is fed to multiple layers.
	// the cost is only calculated once, do it when inpidx is 0
	if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0]; // labels as the first input
        NVMatrix& acts = *_inputs[1];   // activations from the previous layer as the second output
        int numCases = labels.getNumElements(); 
        
		values.clear();
		counts.clear();
		sparse_histogram(labels.getDevData(), numCases, values, counts);
		
		sqrts.resize(_channels, values.size());   // Allocate the matrix for summation calculation. The number of rows is equal to the number of channels. 
                                                  // The number of cols is equal to the number of distinct labels.
		sqrts.apply(NVMatrixOps::Zero());
		// calculate the cost
		float cost = CalculateSqrtSumSquareMatrix(labels, acts, values, counts, sqrts, _channels, acts.getNumRows()/_channels);
		_costv.clear();
		_costv.push_back(cost);
    }
}

void GroupSparsityInLabelCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType){ // For cost layer, the v matrix is not necessary.
	NVMatrix target(*_inputs[1]);
	CalculateGradient(*_inputs[1], *_inputs[0], sqrts, values, counts, _channels, (*_inputs[1]).getNumRows()/_channels, (*_inputs[1]).getNumCols(), target);
	_prev[inpIdx]->getActsGrad().add(target, scaleTargets, -_coeff);
}

// public 
GroupSparsityInLabelCostLayer::GroupSparsityInLabelCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false){

	// Initialize variables from python
	_channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}





















// Define functions that will create the Layers and Neurons defined inside this shared library
Layer* CreateGroupSparsityInLabelCostLayer(ConvNet* convNet, PyObject* paramsDict) {
	return new GroupSparsityInLabelCostLayer(convNet, paramsDict);
}

Neuron* CreateDropoutNeuron(PyObject* neuronParamsDict) {
	return new DropoutNeuron(neuronParamsDict);
}

// The following two functions are exported from the shared object.
// All constructors of the layers or neurons defined in this shared library should be returned from this function.
// Explicitly export this two functions which is used as the standard interface of the plugin.
extern "C" __attribute__((visibility("default")))
std::map<string, layerConFunc> layerConstructor(){
	std::cout<<"Getting the layer constructors inside this shared library"<<std::endl;
	std::map<string, layerConFunc> ret;
	ret["cost.gsinlabel"] = &CreateGroupSparsityInLabelCostLayer;
	return ret;
}

extern "C" __attribute__((visibility("default")))
std::map<string, neuronConFunc> neuronConstructor(){
	std::cout<<"Getting the neuron constructors inside this shared library"<<std::endl;
	std::map<string, neuronConFunc> ret;
	ret["dropout"] = &CreateDropoutNeuron;
	return ret;
}