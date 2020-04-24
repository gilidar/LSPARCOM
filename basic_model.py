from keras.models import Model
from keras.layers import Input, Conv2D, Add, Subtract
from keras.constraints import nonneg, Constraint
import keras
from keras.callbacks import Callback
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import keras.backend as K

def weighted_mse_wL1(yTrue, yPred):
    mse = K.square(yTrue-yPred)
    L1 = K.abs(yPred)
    w_GT = K.abs(yTrue)
    w_mse = K.cast(K.greater(w_GT, 1e-14), 'float32')
    w_L1 = (1-w_mse)
    return K.mean(w_mse*mse + 1*w_L1*L1) # 0.7

class WeightClip(Constraint):
    def __init__(self, mn=0, mx=1.0):
        self.mn = mn
        self.mx = mx

    def __call__(self, p):
        return K.clip(p, self.mn, self.mx)

    def get_config(self):
        return {
            'name': self.__class__.__name__,
            'minimum': self.mn,
            'maximum': self.mx
        }

class RadialConstraint(Constraint):
    def __call__(self, w):
        height, width, _, _ = w.shape
        R = np.array(np.floor(height/2))
        vec = np.arange(-R, R+1)
        X, Y = np.meshgrid(vec, vec)
        CIRC = X**2 + Y**2
        CIRC_VEC = np.reshape(CIRC, [height**2])
        CIRC_VEC_U = np.unique(CIRC_VEC)
        for ind in range(CIRC_VEC_U.shape[0]):
            MAT = np.array((CIRC == CIRC_VEC_U[ind]))
            w_mat = w[MAT == 1]
            val = tf.reduce_mean(tf.squeeze(w_mat), keepdims=True)
            if ind == 0:
                w_new = val*MAT
            else:
                w_new = w_new + val * MAT
        return tf.reshape(w_new, (height, width, 1,1))

def custom_gauss(shape, dtype=None):
    sig = 1
    res = normaG(matlab_style_gauss2D(shape=(shape[1], shape[1]), sigma=sig), 1)
    res = np.expand_dims(res, axis=-1)
    res = np.expand_dims(res, axis=-1)
    Kres = K.variable(res)
    return Kres

def customI(shape, dtype=None):
    ker_size = shape[0]
    mid = int(np.floor(ker_size/2))
    res = np.zeros(shape)
    res[mid, mid, :, :] = 1
    return K.variable(res)

def custom_mean33(shape, dtype=None):
    ker_size = shape[0]
    mid = int(np.floor(ker_size/2))
    res = np.zeros(shape)
    res[mid, mid, :, :] = 1
    res[mid + 1, mid] = 1
    res[mid - 1, mid] = 1
    res[mid, mid + 1] = 1
    res[mid, mid - 1] = 1
    res[mid + 1, mid + 1] = 1
    res[mid + 1, mid - 1] = 1
    res[mid - 1, mid + 1] = 1
    res[mid - 1, mid - 1] = 1
    res = (1/9)*res
    return K.variable(res)

def custom_mean_big(shape, dtype=None):
    num_vox_xy = shape[0] * shape[1]
    num_vox_zz = shape[2]
    res = (1/(num_vox_xy*num_vox_zz))*np.ones(shape)
    return K.variable(res)

def custom_mean(shape, dtype=None):
    num_vox_xy = (shape[0]-2) * (shape[1]-2)
    num_vox_zz = shape[2]
    res = (1/(num_vox_xy*num_vox_zz))*np.ones(shape)
    res[0, :] = 0
    res[shape[0]-1, :] = 0
    res[:, 0] = 0
    res[:, shape[1]-1] = 0
    return K.variable(res)

def normaG(ar, val):
    # Normalize according to known value
    aro = ar - np.amin(ar)
    maxv = np.amax(aro)
    if maxv != 0:
        aro = val*aro/maxv
    return aro

def matlab_style_gauss2D(shape=(7, 7), sigma=1):
    """
    2D gaussian filter - should give the same result as:
    MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h.astype(dtype=K.floatx())
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h * 2.0
    h = h.astype('float64')
    return h


''' (approximate) proximity operator '''
class Prox(Layer):
    def __init__(self, proxtype=3, ialpha=0.95, itau=8, **kwargs):
        super(Prox, self).__init__(**kwargs)
        self.proxtype = proxtype
        self.ialpha = ialpha
        self.itau = itau

    def build(self, input_shape):
        self.alpha = self.add_weight(name='kernel',
                                     shape=(1,),
                                     initializer=keras.initializers.Constant(self.ialpha), #constraint=WeightClip(0, 1.0),  # 0<alpha<1
                                     trainable=True)
        self.tau = self.add_weight(name='kernel',
                                   shape=(1,),
                                   initializer=keras.initializers.Constant(self.itau),
                                   constraint=nonneg(),  # Positive threshold
                                   trainable=True)
        super(Prox, self).build(input_shape)

    def call(self, x):

        a = x.get_shape().as_list()
        L = a[1] * a[2]
        B = tf.shape(x)[0]
        xp = K.permute_dimensions(x, pattern=(0, 3, 1, 2))
        xr = tf.reshape(xp, [a[3] * B, L])
        sortvec = tf.math.top_k(xr, k=L, sorted=True)
        st = sortvec.values[:, int(0.99 * L)]  # 1 percentile [min]
        en = sortvec.values[:, int(0.01 * L)]  # 99 percentile [max]
        th0 = st + (en - st) * self.alpha
        th0_r = tf.reshape(th0, [B, a[3], 1])
        ## Copy and reshape
        th_M = K.tile(th0_r, L)
        th_Mp = K.permute_dimensions(th_M, pattern=(0, 2, 1))
        th = tf.reshape(th_Mp, [B, a[1], a[2], a[3]])

        # Discard negs
        val0 = K.cast(K.greater(th, 1e-14), 'float32')
        th = th * val0  # zero negs
        val1 = 1 - val0
        val_st = th + val1  # For numerical stability
        tau = self.tau * K.ones((B, a[1], a[2], a[3])) / val_st

        #th = self.alpha; tau = self.tau

        if self.proxtype == 0:  # soft thresholding
            return K.sign(x) * K.relu(K.abs(x) - th)
        elif self.proxtype == 1:  # smooth sigmoid based soft thresholding
            return x / (1 + K.exp(-tau * (K.abs(x) - th)))
        elif self.proxtype == 2:  # positive soft threshold
            return K.relu(x - th)
        elif self.proxtype == 3:  # positive smooth sigmoid based soft thresholding
            return K.relu(x) / (1 + K.exp(-tau * (K.abs(x) - th)))

        def compute_output_shape(self, input_shape):
            return input_shape

''' Scaling operator '''
class Scale(Layer):
    def __init__(self, nonegf=1, val=0.1, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.noneg = nonegf
        self.ival = val

    def build(self, input_shape):
        if self.noneg == 0:
            self.alpha = self.add_weight(name='kernel',
                                     shape=(1,),
                                     initializer=keras.initializers.Constant(value=self.ival),
                                     constraint=nonneg(),
                                     trainable=True)
        else:
            self.alpha = self.add_weight(name='kernel',
                                         shape=(1,),
                                         initializer=keras.initializers.Constant(value=self.ival),
                                         constraint=nonneg(),
                                         trainable=False)
        super(Scale, self).build(input_shape)

    def call(self, x):
        return x*self.alpha

    def get_config(self):
        config = {'nonegf': self.noneg, 'val': self.ival}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

''' Build model '''
def buildModel(input_dim, numfolds, upfactor = 4, proxtype=1):
     input_ = Input(shape=(input_dim[0], input_dim[1], 1))
     output_ = Unfolding_model(input_, numfolds, upfactor, proxtype=proxtype)
     model = Model(inputs=input_, outputs=output_)
     return model


''' Design model'''
def Unfolding_model(input, numfolds, upF, proxtype):
    psf_M = 29
    psf_A = 25

    Lfv = Conv2D(1, (psf_A, psf_A), activation=None, padding='same', use_bias=False, strides=[1, 1],
                   kernel_initializer=custom_gauss, kernel_constraint=RadialConstraint(), name='convA')(input)
    xw = Lfv

    for k in range(0, numfolds):
        x_thresh = Prox(proxtype=proxtype, name='x_thresh_{}'.format(k))(xw)
        x_thresh_P1 = Conv2D(1, (psf_M, psf_M), activation=None, padding='same', use_bias=False, strides=[1, 1],
                   kernel_initializer=custom_gauss, kernel_constraint=RadialConstraint(), name='convM_{}'.format(k))(x_thresh)
        x_a0 = Subtract(name='xs_{}'.format(k))([Lfv, x_thresh_P1])
        xw = Add(name='xa_{}'.format(k))([x_a0, x_thresh])

    x_out = Prox(proxtype=proxtype, name='prox_out')(xw)
    x_outs = Scale(nonegf=0, val=0.01, name='out_scale')(x_out) #0.001

    return x_outs




