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
GroupSparsityInLabelCostLayer::GroupSparsityInLabelCostLayer(ConvNet* convNet, PyObject* paramsDict) : CostLayer(convNet, paramsDict, false) {

	// Initialize variables from python
	_channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}






/* 
 * =======================
 * FFCLayer Filter Fully Connected
 * =======================
 */
FFCLayer::FFCLayer(ConvNet* convNet, PyObject* paramsDict) : WeightLayer(convNet, paramsDict, true, false) {
    _wStep = 0.1;
    _bStep = 0.01;
    _channels = pyDictGetInt(paramsDict, "channels");
    _in_nodes = pyDictGetInt(paramsDict, "in_nodes");
    _out_nodes = pyDictGetInt(paramsDict, "out_nodes");
}

void FFCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	// input is pixels*_in_nodes*channels x batch_num
	// output is pixels*_out_nodes*channels x batch_num
	// First reshape the input into channels*in_nodes x batchnum*pixels
	assert(_inputs[inpIdx]->getNumCols() % (_channels*_in_nodes) == 0);
	int pixels = _inputs[inpIdx]->getNumCols() / (_channels*_in_nodes);
	_inputs[inpIdx]->reshape(_inputs[inpIdx]->getNumElements()/(_channels*_in_nodes), (_channels*_in_nodes));
	// prepare the space for the output
	getActs().resize(_inputs[inpIdx]->getNumElements()/(_channels*_in_nodes), (_channels*_out_nodes));
	// batched dot product
	for (int i=0; i<_channels; i++) {
		NVMatrix& in = _inputs[inpIdx]->sliceCols(i*_in_nodes, (i+1)*_in_nodes);
		NVMatrix& out = getActs().sliceCols(i*_out_nodes, (i+1)*_out_nodes);
		NVMatrix& W = (*_weights[inpIdx]).sliceCols(i*_out_nodes, (i+1)*_out_nodes);
		// printf("out size %dx%d in size %dx%d\n", out.getNumRows(), out.getNumCols(), in.getNumRows(), in.getNumCols());
		out.addProduct(in, W, scaleTargets, 1);
		delete &in;
		delete &out;
		delete &W;
	}
    
    if (scaleTargets == 0) {// At the first run, the scaleTarget = 0, add the bias. The bias will be added only once.
    	// printf("getActs size %dx%d bias size %dx%d\n", getActs().getNumRows(), getActs().getNumCols(), _biases->getW().getNumRows(), _biases->getW().getNumCols());
        getActs().addVector(_biases->getW());
    }
    _inputs[inpIdx]->reshape(_inputs[inpIdx]->getNumElements()/(_channels*_in_nodes*pixels), (_channels*_in_nodes*pixels));
    getActs().reshape(getActs().getNumElements()/(_channels*_out_nodes*pixels), (_channels*_out_nodes*pixels));
    /* if (passType == PASS_TEST && this->_name.compare("ffc6")==0) {
      // Get data layer
      Layer& datalayer = this->_convNet->getLayer(0);
      datalayer.getActs().print(datalayer.getActs().getNumRows(), datalayer.getActs().getNumCols());
      printf("\n\n");
      getActs().print(getActs().getNumRows(), getActs().getNumCols());
      exit(0);
      } */
}

void FFCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
	assert(v.getNumCols() % (_channels*_out_nodes) == 0);
	int pixels = v.getNumCols() / (_channels*_out_nodes);

	v.reshape(v.getNumElements()/(_channels*_out_nodes), (_channels*_out_nodes));
	_prev[inpIdx]->getActsGrad().resize(v.getNumElements()/(_channels*_out_nodes), (_channels*_in_nodes));

    for (int i=0; i<_channels; i++) {
		NVMatrix& t = _prev[inpIdx]->getActsGrad().sliceCols(i*_in_nodes, (i+1)*_in_nodes);
		NVMatrix& vi = v.sliceCols(i*_out_nodes, (i+1)*_out_nodes);
		NVMatrix& W = (*_weights[inpIdx]).sliceCols(i*_out_nodes, (i+1)*_out_nodes);
		NVMatrix& weights_T = W.getTranspose();
		// printf("t size %dx%d vi size %dx%d\n", t.getNumRows(), t.getNumCols(), vi.getNumRows(), vi.getNumCols());
		t.addProduct(vi, weights_T, scaleTargets, 1);
		delete &t;
		delete &vi;
		delete &W;
		delete &weights_T;
	}
	v.reshape(v.getNumElements()/(_channels*_out_nodes*pixels), (_channels*_out_nodes*pixels));
    _prev[inpIdx]->getActsGrad().reshape(v.getNumElements()/(_channels*_out_nodes*pixels), (_channels*_in_nodes*pixels));
}

void FFCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) { 
    // Is the numCases the number of samples ? 
    // Seems possible. If so, how does the other layers whose output for a single sample is a matrix pass the matrix of multiple samples?
    // Are they all unfolded to a vector?
	int numCases = v.getNumRows();
	float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;

    assert(v.getNumCols() % (_channels*_out_nodes) == 0);
	int pixels = v.getNumCols() / (_channels*_out_nodes);

	v.reshape(v.getNumElements()/(_channels*_out_nodes), (_channels*_out_nodes));

    _biases->getGrad().addSum(v, 0, 0, scaleBGrad);

    v.reshape(v.getNumElements()/(_channels*pixels*_out_nodes), (_channels*pixels*_out_nodes));
}

void FFCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {

    // Judging from this function and the above one, each row of the matrix v is a sample.
    // Which means that it is quite possible that all the inputs and outputs are unfolded version of the activations.
    int numCases = v.getNumRows();

    float scaleInc = (_weights[inpIdx].getNumUpdates() == 0 && passType != PASS_GC) * _weights[inpIdx].getMom();
    float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;

    assert(v.getNumCols() % (_channels*_out_nodes) == 0);
	int pixels = v.getNumCols() / (_channels*_out_nodes);

	v.reshape(v.getNumElements()/(_channels*_out_nodes), (_channels*_out_nodes));
	_prev[inpIdx]->getActs().reshape(v.getNumElements()/(_channels*_out_nodes), (_channels*_in_nodes));
	_weights[inpIdx].getInc().resize(_in_nodes, _out_nodes*_channels);

	for (int i=0; i<_channels; i++) {
		NVMatrix& prevActs = _prev[inpIdx]->getActs().sliceCols(i*_in_nodes, (i+1)*_in_nodes);
		NVMatrix& prevActs_T = prevActs.getTranspose();
		NVMatrix& vi = v.sliceCols(i*_out_nodes, (i+1)*_out_nodes);
		NVMatrix& W_Inc = _weights[inpIdx].getInc().sliceCols(i*_out_nodes, (i+1)*_out_nodes);
		// printf("W_Inc size %dx%d prevActs_T size %dx%d\n", W_Inc.getNumRows(), W_Inc.getNumCols(), prevActs_T.getNumRows(), prevActs_T.getNumCols());
		W_Inc.addProduct(prevActs_T, vi, scaleInc, scaleGrad);
		delete &prevActs;
		delete &prevActs_T;
		delete &vi;
		delete &W_Inc;
	}

	v.reshape(v.getNumElements()/(_channels*_out_nodes*pixels), (_channels*_out_nodes*pixels));
	_prev[inpIdx]->getActs().reshape(v.getNumElements()/(_channels*_out_nodes*pixels), (_channels*_in_nodes*pixels));
}









// Define functions that will create the Layers and Neurons defined inside this shared library
Layer* CreateGroupSparsityInLabelCostLayer(ConvNet* convNet, PyObject* paramsDict) {
	return new GroupSparsityInLabelCostLayer(convNet, paramsDict);
}

Layer* CreateFFCLayer(ConvNet* convNet, PyObject* paramsDict) {
	return new FFCLayer(convNet, paramsDict);
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
	ret["ffc"] = &CreateFFCLayer;
	return ret;
}

extern "C" __attribute__((visibility("default")))
std::map<string, neuronConFunc> neuronConstructor(){
	std::cout<<"Getting the neuron constructors inside this shared library"<<std::endl;
	std::map<string, neuronConFunc> ret;
	ret["dropout"] = &CreateDropoutNeuron;
	return ret;
}
