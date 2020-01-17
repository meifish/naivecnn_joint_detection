from keras import backend as K

def channel_softmax_2d(x):

    ndim = K.ndim(x)
    if ndim == 4:
        e = K.exp(x - K.max(x, axis=(1,2), keepdims=True))
        s = K.sum(e, axis=(1,2), keepdims=True)
        return e / s
    else:
        raise ValueError('This function is specific for 4D tensors. '
                'Here, ndim=' + str(ndim))
