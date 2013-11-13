# plugin.py


from math import exp
import sys
import ConfigParser as cfg
import os
import numpy as n
import numpy.random as nr
from math import ceil, floor
from ordereddict import OrderedDict
from os import linesep as NL
from options import OptionsParser
import re
from layer import CostParser, LayerParsingError, ParamNeuronParser, WeightLayerParser


class GroupSparsityInLabelCostParser(CostParser):
    def __init__(self):
        CostParser.__init__(self, num_inputs=2)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = CostParser.parse(self, name, mcp, prev_layers, model)
        if dic['numInputs'][0] != 1: # first input must be labels
            raise LayerParsingError("Layer '%s': dimensionality of first input must be 1" % name)
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['imgSize'] = mcp.safe_get_int(name, 'imgSize')

        print "Initialized groupSparsityInLabel cost '%s' with %d channels and %dx%d inputs" % (name, dic['channels'], dic['imgSize'], dic['imgSize'])
        return dic

class FFCLayerParser(WeightLayerParser):
    def __init__(self):
        WeightLayerParser.__init__(self)
        
    def parse(self, name, mcp, prev_layers, model):
        dic = WeightLayerParser.parse(self, name, mcp, prev_layers, model)
        
        dic['usesActs'] = False
        dic['channels'] = mcp.safe_get_int(name, 'channels')
        dic['in_nodes'] = mcp.safe_get_int(name, 'in_nodes')
        dic['out_nodes'] = mcp.safe_get_int(name, 'out_nodes')
	
	pixels = dic['numInputs'][0] / (dic['in_nodes']*dic['channels'])
        dic['outputs'] = dic['out_nodes'] * dic['channels'] * pixels
        
        # assert dic['numInputs'] == [ dic['in_nodes'] * dic['channels'] ] # there is only one input. This input is of size in_nodes*channels

        self.verify_num_range(dic['outputs'], 'outputs', 1, None)
        
        self.make_weights(dic['initW'], [dic['in_nodes']], [dic['out_nodes']*dic['channels']], order='F')
        self.make_biases(1, dic['out_nodes']*dic['channels'], order='F')

        print "Initialized filter-fully-connected layer '%s', producing %d outputs in %d channels" % (name, dic['out_nodes'], dic['channels'])
        return dic


extra_layer_parsers = {'cost.gsinlabel': lambda: GroupSparsityInLabelCostParser(),
                        'ffc': lambda: FFCLayerParser()}

extra_neuron_parsers = sorted([ParamNeuronParser('dropout[prob]', 'f(x) = rand() < prob ? 0:x', uses_acts=False, uses_inputs=False)],
                        key=lambda x:x.type)
