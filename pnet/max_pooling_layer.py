__author__ = 'jiajunshen'


from pnet.layer import Layer
import numpy as np
import amitgroup as ag

@Layer.register('max-pooling-layer')
class MaxPoolingLayer(Layer):

    def __init__(self, shape=(1, 1), strides=(1, 1), settings={}):
        self._shape = shape
        self._strides = strides
        self._settings = settings
        self._extract_info = {}

    def extract(self, X_F):
        X = X_F[0]
        F = X_F[1]
        #X = X.astype(np.float16)
        #if X.ndim == 3:
            #from pnet.cyfuncs import index_map_pooling as poolf
        #else:
        #print(np.any(np.isnan(X)))
        #print(np.all(np.isfinite(X)))
        print ("inside extrating")
        print(X.shape)
        print(X.dtype)
        support_mask = self._settings.get('support_mask')
        relu = self._settings.get("relu", True)
        if support_mask is not None:
            from pnet.cyfuncs import activation_map_pooling as poolf
            #feature_map = poolf(X, support_mask.astype(np.uint8), F, self._shape, self._strides)
        else:
            from pnet.cyfuncs import activation_map_pooling as poolf
            feature_map_list = []
            batch_number = 500
            batch_size = X.shape[0] // batch_number
            for i in range(batch_number):
                #print i
                feature_map_list.append(poolf(X[i * batch_size : min((i+1) * batch_size, X.shape[0])].astype(np.float32), F, self._shape, self._strides, relu).astype(np.float16))
            feature_map = np.vstack(feature_map_list)
            feature_map.astype(np.float16)

        self._extract_info['concentration'] = np.apply_over_axes(np.mean, feature_map, [0, 1, 2])[0,0,0]
        ag.info("finish pooling")
        #print(feature_map.shape)
        print(np.any(np.isnan(feature_map)))
        print(np.all(np.isfinite(feature_map)))
        return feature_map

    def infoplot(self, vz):
        import pylab as plt

        if 'concentration' in self._extract_info:
            plt.figure(figsize=(6, 3))
            plt.plot(self._extract_info['concentration'], label='concentration')
            plt.savefig(vz.generate_filename(ext='svg'))

            plt.figure(figsize=(4, 4))
            cc = self._extract_info['concentration']
            plt.hist(np.log10(cc[cc>0]), normed=True)
            plt.xlabel('Concentration (10^x)')
            plt.title('Concentration')
            plt.savefig(vz.generate_filename(ext='svg'))

            #vz.log('concentration', self._extract_info['concentration'])
            vz.log('mean concentration: {:.1f}%'.format(100*self._extract_info['concentration'].mean()))


    def save_to_dict(self):
        d = {}
        d['shape'] = self._shape
        d['strides'] = self._strides
        return d

    @classmethod
    def load_from_dict(cls, d):
        obj = cls(d['shape'], d['strides'])
        return obj

    def __repr__(self):
        return 'PoolingLayer(shape={shape}, strides={strides}, has_support_mask={has_support_mask})'.format(
                    shape=self._shape,
                    strides=self._strides,
                    has_support_mask='support_mask' in self._settings,
                    settings=self._settings)
