import numpy as np
import tensorflow as tf

def ortho_init(shape, dtype, partition_info=None):
    #lasagne ortho init for tf
    shape = tuple(shape)
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4: # assumes NHWC
        flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
        raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]].astype(np.float32)

counters = {}
def get_name(layer_name):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def add_checkpoint(x):
    ''' hint that this input should be remembered by our memory efficient gradient implementation '''
    if isinstance(x, tuple) or isinstance(x, list):
        for xi in x:
            add_checkpoint(xi)
    else:
        tf.add_to_collection('checkpoints', x)

# Layer base class, overwrite methods to do something specific
class Layer:
    def __init__(self):
        pass

    def apply(self, x, **kwargs):
        raise NotImplementedError

    def apply_inverse(self, x, **kwargs):
        raise NotImplementedError

    def __call__(self, x, **kwargs):
        if hasattr(self, 'sublayer'):
            return self.sublayer(x, **kwargs)
        else:
            return self.apply(x, **kwargs)

    def invert(self, x, **kwargs):
        if hasattr(self, 'sublayer'):
            return self.sublayer.invert(x, **kwargs)
        else:
            raise NotImplementedError

    def __add__(self, other): # not commutative!
        return AddLayer(self, other)

    def __iadd__(self, other): # just for notational convenience, we don't do anything in place!
        return self.__add__(other)

class AddLayer(Layer):
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __call__(self, x, **kwargs):
        l1 = self.layer1(x, **kwargs)
        if isinstance(self.layer2, Layer):
            out = self.layer2(l1, **kwargs)
        elif isinstance(self.layer2, float) or isinstance(self.layer2, int) or isinstance(self.layer2, tf.Tensor) or isinstance(self.layer2, tf.Variable):
            if isinstance(l1, tuple):
                out = (l1[0] + self.layer2, l1[1])
            else:
                out = l1 + self.layer2
        else:
            raise ('unsupported Layer addition type ' + str(type(self.layer2)))
        return out

    def invert(self, x, **kwargs):
        if isinstance(self.layer2, Layer):
            out = self.layer2.invert(x, **kwargs)
        elif isinstance(self.layer2, float) or isinstance(self.layer2, int) or isinstance(self.layer2, tf.Tensor) or isinstance(self.layer2, tf.Variable):
            if isinstance(x, tuple):
                out = (x[0] - self.layer2, x[1])
            else:
                out = x - self.layer2
        else:
            raise ('unsupported Layer addition type ' + str(type(self.layer2)))
        out = self.layer1.invert(out, **kwargs)
        return out

class Dense(Layer):
    def __init__(self, num_in, num_out, use_bias=True, use_weight_norm=True, name=None, init_scale=1.):
        self.use_bias = use_bias
        if name is None:
            name = get_name('dense')
        self.name = name
        with tf.variable_scope(name):
            self.W = tf.get_variable('W', [num_in,num_out], tf.float32, ortho_init, trainable=True)
            if use_bias:
                self.b = tf.get_variable('b', initializer=tf.zeros(num_out,dtype=tf.float32), trainable=True)
            if use_weight_norm:
                self.g = tf.get_variable('g', initializer=init_scale*np.ones(num_out, dtype=np.float32), trainable=True)
                self.W = self.g * (self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W),axis=0)))

    def apply(self, x, **kwargs):
        y = tf.matmul(x, self.W)
        if self.use_bias:
            y = tf.nn.bias_add(y,self.b)
        return y

class WavenetSeparableConv2d(Layer):
    def __init__(self, num_filters_in, channel_multiplier=1, filter_size=[3,3], stride=[1,1], delay=2, pad='SAME',
                 use_bias=True, use_weight_norm=True, name=None, init_scale=1.):
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.use_bias = use_bias
        self.delay = delay
        self.channel_multiplier = channel_multiplier
        if name is None:
            name = get_name('SeparableWavenetConv2d')
        self.name = name
        with tf.variable_scope(name):
            nout = channel_multiplier * num_filters_in
            self.W1_depth = tf.get_variable('W1_depth', filter_size+[num_filters_in, channel_multiplier], tf.float32, ortho_init, trainable=True)
            self.W1_point = tf.get_variable('W1_point', [1,1,nout,nout], tf.float32, ortho_init, trainable=True)
            self.W2_depth = tf.get_variable('W2_depth', filter_size + [num_filters_in, channel_multiplier], tf.float32, ortho_init, trainable=True)
            self.W2_point = tf.get_variable('W2_point', [1,1,nout,nout], tf.float32, ortho_init, trainable=True)
            if use_weight_norm:
                W1_mat = tf.matmul(tf.reshape(self.W1_depth, [-1, nout]), tf.squeeze(self.W1_point))
                W2_mat = tf.matmul(tf.reshape(self.W2_depth, [-1, nout]), tf.squeeze(self.W2_point))
                self.g1 = tf.get_variable('g1', initializer=init_scale*np.ones(nout, dtype=np.float32), trainable=True)
                self.W1_point = self.g1 * (self.W1_point / tf.sqrt(tf.reduce_sum(tf.square(W1_mat), axis=0)))
                self.g2 = tf.get_variable('g2', initializer=0.1*init_scale*np.ones(nout, dtype=np.float32), trainable=True)
                self.W2_point = self.g2 * (self.W2_point / tf.sqrt(tf.reduce_sum(tf.square(W2_mat), axis=0)))
            if use_bias:
                self.b = tf.get_variable('b', initializer=tf.zeros(nout,dtype=tf.float32), trainable=True)

    def apply(self, x, dones=False, **kwargs):

        T,N,H,W,C = x.get_shape().as_list()
        if T==1:
            x_lagged = tf.Variable(np.zeros((self.delay+1, N, H, W, C), dtype=np.float32), trainable=False)
            # mask and shift forward
            m = 1. - tf.reshape(tf.to_float(dones), [1,N,1,1,1])
            new_x_lagged = m * x_lagged
            x = x_lagged.assign(tf.concat([new_x_lagged[1:], x], axis=0))
        else:
            assert T>self.delay

        x_end = tf.reshape(x[self.delay:], [-1, H, W, C])
        x_start = tf.reshape(x[:-self.delay], [-1, H, W, C])

        y = tf.nn.separable_conv2d(x_end, self.W1_depth, self.W1_point, [1] + self.stride + [1], self.pad)
        _, H, W, C = y.get_shape().as_list() # new shapes
        if self.use_bias:
            y = tf.nn.bias_add(y,self.b)
        y = tf.reshape(y, [-1, N, H, W, C])

        z = tf.reshape(tf.nn.separable_conv2d(x_start, self.W2_depth, self.W2_point, [1] + self.stride + [1], self.pad), [-1, N, H, W, C])

        # mask if training
        if T>1:
            T_out = T - self.delay
            cumsum_dones = tf.cumsum(dones, axis=0)
            mask = 1. - tf.reshape(tf.to_float(tf.greater(cumsum_dones[-T_out:] - cumsum_dones[(-T_out-self.delay):-self.delay],0)), [T_out, N, 1, 1, 1])
            z *= mask

        return y+z

class Conv2d(Layer):
    def __init__(self, num_filters_in, num_filters_out, filter_size=[3,3], stride=[1,1], pad='SAME', use_bias=True, use_weight_norm=True, name=None, init_scale=1.):
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.use_bias = use_bias
        if name is None:
            name = get_name('conv2d')
        self.name = name
        with tf.variable_scope(name):
            self.W = tf.get_variable('W', filter_size+[num_filters_in,num_filters_out], tf.float32, ortho_init, trainable=True)
            if use_bias:
                self.b = tf.get_variable('b', initializer=tf.zeros(num_filters_out,dtype=tf.float32), trainable=True)
            if use_weight_norm:
                self.g = tf.get_variable('g', initializer=init_scale*np.ones(num_filters_out, dtype=np.float32), trainable=True)
                self.W = self.g * (self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W),axis=[0,1,2])))

    def apply(self, x, **kwargs):
        y = tf.nn.conv2d(x, self.W, [1]+self.stride+[1], self.pad)
        if self.use_bias:
            y = tf.nn.bias_add(y,self.b)
        return y

class Conv3d(Layer):
    def __init__(self, num_filters_in, num_filters_out, filter_size=[3,3,3], stride=[1,1,1], pad='SAME', use_bias=True, name=None):
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.use_bias = use_bias
        self.num_filters_out = num_filters_out
        if name is None:
            name = get_name('conv3d')
        self.name = name
        with tf.variable_scope(name):
            self.W = tf.get_variable('W', filter_size+[num_filters_in,num_filters_out], tf.float32, ortho_init, trainable=True)
            if use_bias:
                self.b = tf.get_variable('b', initializer=tf.zeros(num_filters_out,dtype=tf.float32), trainable=True)

    def apply(self, x, **kwargs):
        y = tf.nn.conv3d(x, self.W, [1]+self.stride+[1], self.pad)
        if self.use_bias:
            y += tf.reshape(self.b, [1,1,1,1,self.num_filters_out])
        return y

class Scale(Layer):
    def __init__(self, num_features, name=None):
        self.num_features = num_features
        if name is None:
            name = get_name('scale')
        self.name = name
        with tf.variable_scope(name):
            self.scale = tf.get_variable('scale', initializer=tf.ones(num_features,dtype=tf.float32), trainable=True)

    def apply(self, x, **kwargs):
        return x * tf.reshape(self.scale,[1]*(len(x.get_shape())-1)+[self.num_features])

class ScaleAndShift(Layer):
    def __init__(self, num_features, name=None):
        self.num_features = num_features
        if name is None:
            name = get_name('scale_and_shift')
        self.name = name
        with tf.variable_scope(name):
            self.scale = tf.exp(tf.get_variable('scale', initializer=tf.zeros(num_features,dtype=tf.float32), trainable=True))
            self.shift = tf.get_variable('shift', initializer=tf.zeros(num_features, dtype=tf.float32), trainable=True)

    def apply(self, x, **kwargs):
        return (x + tf.reshape(self.shift,[1]*(len(x.get_shape())-1)+[self.num_features])) * tf.reshape(self.scale,[1]*(len(x.get_shape())-1)+[self.num_features])

class Reshape(Layer):
    def __init__(self, output_shape, input_shape=None):
        self.output_shape = output_shape
        self.input_shape = input_shape
    def _reshape(self, x, s):
        if isinstance(x, tuple):
            return (tf.reshape(x[0], s), x[1])
        else:
            return
    def apply(self, x, **kwargs):
        output_shape = [kwargs[s] if isinstance(s,str) else s for s in self.output_shape]
        return tf.reshape(x, output_shape)
    def invert(self, x, **kwargs):
        assert self.input_shape is not None
        return tf.reshape(x, self.input_shape)

# borrowed from dpkingma
def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1: return x
    n, height, width, n_channels = x.get_shape().as_list()
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor, width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width//factor, n_channels*factor*factor])
    return x

# borrowed from dpkingma
def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1: return x
    n, height, width, n_channels = x.get_shape().as_list()
    assert n_channels >= 4 and n_channels%4 == 0
    x = tf.reshape(x, (-1, height, width, int(n_channels/factor**2), factor, factor))
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height*factor), int(width*factor), int(n_channels/factor**2)))
    return x

class Squeeze2d(Layer):
    def apply(self, x, **kwargs):
        if isinstance(x, tuple):
            return squeeze2d(x[0]), x[1]
        else:
            return squeeze2d(x)
    def invert(self, x, **kwargs):
        if isinstance(x, tuple):
            return unsqueeze2d(x[0]), x[1]
        else:
            return unsqueeze2d(x)

class Transpose(Layer):
    def __init__(self, perm):
        self.perm = tuple(perm)
    def apply(self, x, **kwargs):
        return tf.transpose(x, self.perm)

class BatchNorm(Layer):
    def __init__(self, num_features, rho=0.9, use_offset=True, use_scale=True, name=None):
        self.num_features = num_features
        self.rho = rho
        self.use_offset = use_offset
        self.use_scale = use_scale
        if name is None:
            name = get_name('batch_norm')
        self.name = name
        with tf.variable_scope(name):
            self.avg_batch_mean = tf.Variable(tf.zeros(num_features, dtype=tf.float32),trainable=False)
            self.avg_batch_var = tf.Variable(tf.ones(num_features, dtype=tf.float32), trainable=False)
            if self.use_offset:
                self.offset = tf.Variable(tf.zeros(num_features, dtype=tf.float32), trainable=True)
            else:
                self.offset = None
            if self.use_scale:
                self.scale = tf.Variable(tf.ones(num_features, dtype=tf.float32), trainable=True)
            else:
                self.scale = None

    def __call__(self, x, is_training=True, update_bn_stats=True, bn_block_mode='all_blocks', bn_nr_blocks='all', **kwargs):
        # calc statistics
        if is_training:
            if isinstance(x,list):
                if bn_nr_blocks == 'all':
                    bn_nr_blocks = len(x)
                moms = [tf.nn.moments(xi,tuple([i for i in range(len(xi.get_shape())-1)])) for xi in x[:bn_nr_blocks]]
                mt = sum([x[0] for x in moms]) / len(moms)
                vt = (sum([x[1] for x in moms]) + sum([tf.square(x[0] - mt) for x in moms])) / len(moms)

                if update_bn_stats:
                    bn_stats_ops = [self.avg_batch_mean.assign(self.rho * self.avg_batch_mean + (1. - self.rho) * mt),
                                    self.avg_batch_var.assign(self.rho * self.avg_batch_var + (1. - self.rho) * vt)]
                else:
                    bn_stats_ops = None

                with tf.control_dependencies(bn_stats_ops):
                    if bn_block_mode == 'per_block':
                        return [self.apply(x[i], moms[i][0], moms[i][1], **kwargs) for i in range(len(x))]
                    else:
                        return [self.apply(x[i], mt, vt, **kwargs) for i in range(len(x))]

            else:
                mt, vt = tf.nn.moments(x,tuple([i for i in range(len(x.get_shape())-1)]))
                if update_bn_stats:
                    bn_stats_ops = [self.avg_batch_mean.assign(self.rho * self.avg_batch_mean + (1. - self.rho) * mt),
                                    self.avg_batch_var.assign(self.rho * self.avg_batch_var + (1. - self.rho) * vt)]
                else:
                    bn_stats_ops = None

                with tf.control_dependencies(bn_stats_ops):
                    return self.apply(x, mt, vt, **kwargs)

        else:
            if isinstance(x,list):
                return [self.apply(xi, self.avg_batch_mean, self.avg_batch_var, **kwargs) for xi in x]
            else:
                return self.apply(x, self.avg_batch_mean, self.avg_batch_var, **kwargs)

    def apply(self, x, m, v, **kwargs):
        return tf.nn.batch_normalization(x,m,v,self.offset,self.scale,1e-5)

class ReLU(Layer):
    def apply(self, x, **kwargs):
        return tf.nn.relu(x)

class CReLU(Layer):
    def apply(self, x, **kwargs):
        return tf.concat([tf.nn.relu(x), tf.nn.relu(-x)], axis=-1)

class GroupNorm2d(Layer):
    def __init__(self, C, G=None, eps=1e-10, name=None):
        if G is None:
            G = int(np.ceil(np.sqrt(C)))
            while not C%G == 0:
                G -= 1
        self.G = G
        self.eps = eps
        if name is None:
            name = get_name('GroupNorm2d')
        self.name = name
        with tf.variable_scope(name):
            self.gamma = tf.get_variable(name + '_gamma', shape=[C], dtype=tf.float32, initializer=tf.ones_initializer)
            self.beta = tf.get_variable(name + '_beta', shape=[C], dtype=tf.float32, initializer=tf.zeros_initializer)

    def apply(self, x, **kwargs):
        N, C = x.shape
        x = tf.reshape(x, [N, self.G, C // self.G])
        mean, var = tf.nn.moments(x, 1, keep_dims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, C])
        return x * self.gamma + self.beta

class GroupNorm4d(Layer):
    def __init__(self, C, G=None, eps=1e-10, name=None):
        if G is None:
            G = int(np.ceil(np.sqrt(C)))
            while not C%G == 0:
                G -= 1
        self.G = G
        self.eps = eps
        if name is None:
            name = get_name('GroupNorm4d')
        self.name = name
        with tf.variable_scope(name):
            self.gamma = tf.get_variable(name + '_gamma', shape=[C], dtype=tf.float32, initializer=tf.ones_initializer)
            self.beta = tf.get_variable(name + '_beta', shape=[C], dtype=tf.float32, initializer=tf.zeros_initializer)

    def apply(self, x, **kwargs):
        N, H, W, C = x.shape
        x = tf.reshape(x, [N, H, W, self.G, C // self.G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta

class AvgPool(Layer):
    def __init__(self, ksize=[2,2], padding='VALID'):
        self.ksize = ksize
        self.padding = padding
    def apply(self, x, **kwargs):
        return tf.nn.avg_pool(x,[1]+self.ksize+[1],[1]+self.ksize+[1],self.padding)

class AvgPool3d(Layer):
    def __init__(self, ksize=[2,2,2], padding='VALID'):
        self.ksize = ksize
        self.padding = padding
    def apply(self, x, **kwargs):
        return tf.nn.avg_pool3d(x,[1]+self.ksize+[1],[1]+self.ksize+[1],self.padding)

class GlobalAvgPool(Layer):
    def apply(self, x, **kwargs):
        return tf.reduce_mean(x,[i for i in range(1,len(x.get_shape())-1)])

class GlobalMaxPool(Layer):
    def apply(self, x, **kwargs):
        return tf.reduce_max(x,[i for i in range(1,len(x.get_shape())-1)])

class Dropout(Layer):
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
    def apply(self, x, is_training=True, **kwargs):
        if is_training and self.drop_prob>0:
            x = tf.nn.dropout(x, 1. - self.drop_prob, seed=np.random.randint(99999999999))
        return x

class Resnet(Layer):
    def __init__(self, num_features_in, num_features_hidden, nonlinearity=ReLU):
        self.num_features_in = num_features_in
        self.num_features_hidden = num_features_hidden
        self.nonlinearity = nonlinearity
        self.residual_layer = BatchNorm(num_features_in) + nonlinearity() + Conv2d(num_features_in, num_features_hidden, use_bias=False) + \
                              BatchNorm(num_features_hidden) + nonlinearity() + Conv2d(num_features_hidden, num_features_in, use_bias=False)

    def __call__(self, x, **kwargs):
        r = self.residual_layer(x, **kwargs)
        out = x+r
        add_checkpoint(out)
        return out

class Resnet3d(Resnet):
    def __init__(self, num_features_in, num_features_hidden, nonlinearity=ReLU):
        self.num_features_in = num_features_in
        self.num_features_hidden = num_features_hidden
        self.nonlinearity = nonlinearity
        self.residual_layer = BatchNorm(num_features_in) + nonlinearity() + Conv3d(num_features_in, num_features_hidden, use_bias=False) + \
                              BatchNorm(num_features_hidden) + nonlinearity() + Conv3d(num_features_hidden, num_features_in, use_bias=False)

def my_concat(xs, axis):
    return xs
    #min_dim = 999999 * np.ones(len(xs[0].get_shape().as_list())-2, dtype=np.int16)
    #for x in xs:
    #    min_dim = np.minimum(min_dim, x.get_shape().as_list()[1:-1])
    #out = []
    #for x in xs:
    #    xd = (np.array(x.get_shape().as_list()[1:-1]) - min_dim) // 2
    #    if any(list(xd > 0)):
    #        if len(xd) == 1:
    #            out.append(x[:, xd[0]:-xd[0], :])
    #        elif len(xd) == 2:
    #            out.append(x[:, xd[0]:-xd[0], xd[1]:-xd[1], :])
    #        else:
    #            out.append(x[:,xd[0]:-xd[0],xd[1]:-xd[1],xd[2]:-xd[2],:])
    #return tf.concat(out, axis)

class Densenet(Layer):
    def __init__(self, num_features_in, num_features_out, drop_prob=0.):
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU() + Conv2d(num_features_in, num_features_out, use_bias=(drop_prob>0)) \
                         + Dropout(drop_prob) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

    def __call__(self, x, **kwargs):
        l = self.sub_layer(x, **kwargs)
        add_checkpoint(l)
        if isinstance(x,list):
            out = []
            for xi,li in zip(x,l):
                if isinstance(xi, tuple):
                    out.append(my_concat(xi + (li,),axis=len(li.get_shape().as_list())-1))
                else:
                    out.append(my_concat((xi, li),axis=len(li.get_shape().as_list())-1))
            return out
        else:
            if isinstance(x, tuple):
                return my_concat(x + (l,),axis=len(l.get_shape().as_list())-1)
            else:
                return my_concat((x,l),axis=len(l.get_shape().as_list())-1)

class DensenetB(Densenet):
    def __init__(self, num_features_in, num_features_out, num_features_hidden=None, drop_prob=0.):
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        if num_features_hidden is None:
            num_features_hidden = num_features_out*4
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU() + Conv2d(num_features_in, num_features_hidden, [1,1], use_bias=False) + BatchNorm(num_features_hidden) + \
                         ReLU() + Conv2d(num_features_hidden, num_features_out, use_bias=(drop_prob>0)) + Dropout(drop_prob) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

class Densenet3d(Densenet):
    def __init__(self, num_features_in, num_features_out, drop_prob=0.):
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU() + Conv3d(num_features_in, num_features_out, use_bias=(drop_prob>0)) \
                         + Dropout(drop_prob) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

class DensenetB3d(Densenet):
    def __init__(self, num_features_in, num_features_out, num_features_hidden=None, drop_prob=0.):
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        if num_features_hidden is None:
            num_features_hidden = num_features_out * 4
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU() + Conv3d(num_features_in, num_features_hidden, [1,1,1], use_bias=False) + BatchNorm(num_features_hidden) + \
                         ReLU() + Conv3d(num_features_hidden, num_features_out, use_bias=(drop_prob>0)) + Dropout(drop_prob) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

class Densenet3dFactored(Densenet):
    def __init__(self, num_features_in, num_features_out, zyx_dim, drop_prob=0.):
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU()
        if num_features_in >= 8*num_features_out:
            num_features_hidden = 4*num_features_out
            self.sub_layer += Conv3d(num_features_in, num_features_hidden, [1,1,1], use_bias=False) + BatchNorm(num_features_hidden) + ReLU()
        else:
            num_features_hidden = num_features_in
        self.sub_layer += Reshape([-1,zyx_dim[1],zyx_dim[2],num_features_hidden]) + Conv2d(num_features_hidden, num_features_out, use_bias=False) \
                          + BatchNorm(num_features_out) + ReLU() + Reshape([-1,zyx_dim[0],zyx_dim[1],zyx_dim[2],num_features_out]) \
                          + Conv3d(num_features_out, num_features_out, [3,1,1], use_bias=(drop_prob>0)) + Dropout(drop_prob) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

class Transition(Layer):
    def __init__(self, num_features_in, num_features_out=None, drop_prob=0.):
        if num_features_out is None:
            num_features_out = num_features_in // 2
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU() + Conv2d(num_features_in, num_features_out, [1,1], use_bias=(drop_prob>0)) \
                         + Dropout(drop_prob) + AvgPool([2,2]) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

    def __call__(self, x, **kwargs):
        out = self.sub_layer(x, **kwargs)
        add_checkpoint(out)
        return out

class Transition3d(Transition):
    def __init__(self, num_features_in, num_features_out=None, drop_prob=0.):
        if num_features_out is None:
            num_features_out = num_features_in // 2
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.drop_prob = drop_prob
        self.sub_layer = ScaleAndShift(num_features_in) + ReLU() + Conv3d(num_features_in, num_features_out, [1,1,1], use_bias=(drop_prob>0)) \
                         + Dropout(drop_prob) + AvgPool3d([2,2,2]) + BatchNorm(num_features_out,use_offset=False,use_scale=False)

def my_matmul(a, b):
    s = a.get_shape().as_list()[:-1]
    bx,by = b.get_shape().as_list()
    y = tf.matmul(tf.reshape(a,[-1,bx]), b)
    return tf.reshape(y, s+[by])
    #return tf.matmul(a, tf.expand_dims(tf.expand_dims(b, axis=0),axis=0))

class Invertible1x1Conv(Layer):
    def __init__(self, nchannels, name=None):
        w_init = np.linalg.qr(np.random.randn(nchannels, nchannels), mode='complete')[0]
        if name is None:
            name = get_name('Invertible1x1Conv')
        self.name = name
        with tf.variable_scope(name):
            self.W = tf.get_variable("W", dtype=tf.float32, initializer=w_init.astype('float32'))
            self.W_inv = tf.matrix_inverse(self.W) # is this numerically stable??
            self.b = tf.get_variable("b", dtype=tf.float32, initializer=np.zeros(nchannels, dtype=np.float32))
        self.log_abs_det = tf.cast(tf.log(abs(tf.matrix_determinant(tf.cast(self.W, 'float64')))), 'float32') # is this numerically stable??

    def __call__(self, x, **kwargs):
        if isinstance(x, tuple):
            x_in, logdet = x
        else:
            x_in = x
            logdet = 0.
        N,H,W,C = x_in.get_shape().as_list()
        x_out = my_matmul(x_in, self.W)+self.b
        logdet += H*W*self.log_abs_det
        return x_out, logdet

    def invert(self, x, **kwargs):
        if isinstance(x, tuple):
            x_out, logdet = x
        else:
            x_out = x
            logdet = 0.
        N,H,W,C = x_out.get_shape().as_list()
        x_in = my_matmul(x_out-self.b, self.W_inv)
        logdet -= H*W*self.log_abs_det
        return x_in, logdet

class StableInvertible1x1Conv(Layer):
    ''' numerically stable version, using LDUP decomposition '''
    def __init__(self, nchannels, name=None):
        p = np.linalg.qr(np.random.randn(nchannels, nchannels))[0]
        if name is None:
            name = get_name('StableInvertible1x1Conv')
        self.name = name
        with tf.variable_scope(name):
            self.P = tf.constant(p.astype(np.float32))
            self.lmask = tf.constant(np.tril(np.ones((nchannels, nchannels), dtype=np.float32),-1))
            self.umask = tf.constant(np.tril(np.ones((nchannels, nchannels), dtype=np.float32),-1).T)
            self.l = tf.get_variable('l', [nchannels,nchannels], dtype=tf.float32, initializer=tf.zeros_initializer)
            self.u = tf.get_variable('u', [nchannels,nchannels], dtype=tf.float32, initializer=tf.zeros_initializer)
            self.d_log_diag = tf.get_variable('d_log_diag', [nchannels], tf.float32, initializer=tf.zeros_initializer)
            eye = tf.constant(np.eye(nchannels, dtype=np.float32))
            self.L = self.lmask * self.l + eye
            self.L_inv = tf.matrix_inverse(self.L)
            self.U = self.umask * self.u + eye
            self.U_inv = tf.matrix_inverse(self.U)
            self.D = tf.exp(self.d_log_diag)
            self.D_inv = tf.exp(-self.d_log_diag)
            self.W = tf.matmul(self.L, tf.reshape(self.D,[-1,1]) * tf.matmul(self.U, self.P))
            self.W_inv = tf.matmul(tf.transpose(self.P, [1,0]), tf.matmul(self.U_inv * tf.reshape(self.D_inv, [1,-1]), self.L_inv))
            self.b = tf.get_variable("b", dtype=tf.float32, initializer=np.zeros(nchannels, dtype=np.float32))
        self.log_abs_det = tf.reduce_sum(self.d_log_diag)

    def __call__(self, x, **kwargs):
        if isinstance(x, tuple):
            x_in, logdet = x
        else:
            x_in = x
            logdet = 0.
        N,H,W,C = x_in.get_shape().as_list()
        x_out = my_matmul(x_in, self.W)+self.b
        logdet += H*W*self.log_abs_det
        return x_out, logdet

    def invert(self, x, **kwargs):
        if isinstance(x, tuple):
            x_out, logdet = x
        else:
            x_out = x
            logdet = 0.
        N,H,W,C = x_out.get_shape().as_list()
        x_out -= self.b
        x_in = my_matmul(x_out, self.W_inv)
        logdet -= H*W*self.log_abs_det
        return x_in, logdet

class Revnet(Layer):
    def __init__(self, num_features_in, num_features_hidden):
        self.num_features_in = num_features_in
        self.num_features_hidden = num_features_hidden
        self.residual_layer1 = CReLU() + Conv2d(num_features_in, num_features_hidden) + \
                               ReLU() + Conv2d(num_features_hidden, num_features_in//2, init_scale=0.)
        self.residual_layer2 = CReLU() + Conv2d(num_features_in, num_features_hidden) + \
                               ReLU() + Conv2d(num_features_hidden, num_features_in//2, init_scale=0.)

    def __call__(self, x, **kwargs):
        if isinstance(x, tuple):
            x_in, logdet = x
        else:
            x_in = x
            logdet = 0.

        xa, xb = tf.split(x_in, 2, axis=-1)

        xb += self.residual_layer1(xa)
        xa += self.residual_layer2(xb)

        x_out = tf.concat([xa,xb],axis=-1)
        return x_out, logdet

    def invert(self, x, **kwargs):
        if isinstance(x, tuple):
            x_in, logdet = x
        else:
            x_in = x
            logdet = 0.

        xa, xb = tf.split(x_in, 2, axis=-1)

        xa -= self.residual_layer2(xb)
        xb -= self.residual_layer1(xa)

        x_out = tf.concat([xa, xb], axis=-1)
        return x_out, logdet

def elu_plus_one(x):
    return tf.where(x>0, x+1., tf.exp(x))

def log_elu_plus_one(x):
    return tf.where(x>0, tf.log(1.+x), x)

class RealNVP(Layer):
    def __init__(self, num_features_in, num_features_hidden):
        self.num_features_in = num_features_in
        self.num_features_hidden = num_features_hidden
        self.residual_layer1 = GroupNorm4d(num_features_in//2) + CReLU() + Conv2d(num_features_in, num_features_hidden) + \
                               ReLU() + Conv2d(num_features_hidden, num_features_in, init_scale=0.)
        self.residual_layer2 = GroupNorm4d(num_features_in//2) + CReLU() + Conv2d(num_features_in, num_features_hidden) + \
                               ReLU() + Conv2d(num_features_hidden, num_features_in, init_scale=0.)

    def __call__(self, x, **kwargs):
        if isinstance(x, tuple):
            x_in, logdet = x
        else:
            x_in = x
            logdet = 0.

        xa, xb = tf.split(x_in, 2, axis=-1)

        rb, lsb = tf.split(self.residual_layer1(xa), 2, axis=-1)
        lsb *= 0.01
        xb = xb * elu_plus_one(lsb) + rb
        logdet += tf.reduce_sum(log_elu_plus_one(lsb), axis=[1,2,3])

        ra, lsa = tf.split(self.residual_layer2(xb), 2, axis=-1)
        lsa *= 0.01
        xa = xa * elu_plus_one(lsa) + ra
        logdet += tf.reduce_sum(log_elu_plus_one(lsa), axis=[1, 2, 3])

        x_out = tf.concat([xa,xb],axis=-1)
        return x_out, logdet

    def invert(self, x, **kwargs):
        if isinstance(x, tuple):
            x_in, logdet = x
        else:
            x_in = x
            logdet = 0.

        xa, xb = tf.split(x_in, 2, axis=-1)

        ra, lsa = tf.split(self.residual_layer2(xb), 2, axis=-1)
        lsa *= 0.01
        xa = (xa - ra) / elu_plus_one(lsa)
        logdet -= tf.reduce_sum(log_elu_plus_one(lsa), axis=[1, 2, 3])

        rb, lsb = tf.split(self.residual_layer1(xa), 2, axis=-1)
        lsb *= 0.01
        xb = (xb - rb) / elu_plus_one(lsb)
        logdet -= tf.reduce_sum(log_elu_plus_one(lsb), axis=[1, 2, 3])

        x_out = tf.concat([xa, xb], axis=-1)
        return x_out, logdet
