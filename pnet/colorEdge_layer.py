from __future__ import division, print_function, absolute_import 

import amitgroup as ag
import numpy as np
from pnet.layer import Layer

@Layer.register('color-edge-layer')
class ColorEdgeLayer(Layer):
    def __init__(self, **kwargs):
        self._edge_settings = kwargs 

    def extract(self, X):
        assert X.ndim == 4
        channels = X.shape[-1]
        edges = ag.features.bedges(np.mean(X,axis = -1), **self._edge_settings)
        colorFeature = ag.features.colorEdges(X) 
        result = np.concatenate((edges,colorFeature), axis = -1)
        return result

    def save_to_dict(self):
        d = {}
        d['edge_settings'] = self._edge_settings
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(**d['edge_settings'])       
        return obj

