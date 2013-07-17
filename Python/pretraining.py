# this code contains a initWFunc function that will load the pretrained weights from another model, and use the parameter from the model to initilize the current model.

from gpumodel import IGPUModel

class initWError(Exception):
    pass

def initWFrom(name, idx, shape, params=None):
	assert(params != None)
	assert(len(params) > 0)
	assert(len(params) > idx)
	(checkPointFile, layerName) = params[idx].split('.')

	net = IGPUModel.load_checkpoint(checkPointFile)
	layernames = [ layer['name'] for layer in net['model_state']['layers'] ]
	if not layerName in layernames:
		raise initWError("There is layer named '%s' in file '%s'" % (layerName, checkPointFile))
	else:
		weightlist = net['model_state']['layers'][layernames.index(layerName)]['weights']
		assert(len(weightlist) > idx)
		assert(weightlist[idx].shape == shape)
		return weightlist[idx]
