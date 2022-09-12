import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import layers

class rc_reg(regularizers.Regularizer):
    def __init__(self, num_filters, _lambda=5e-4, eta=1e-3, gamma=5, tau=1e-4):
        self.num_filters = num_filters
        self._lambda = _lambda
        self.eta = eta
        self.gamma = gamma
        self.tau = tau

    def __call__(self, x):        
        return self._lambda * tf.norm(x) + self.eta * self.rs_loss(x) + self.gamma * self.rc_loss(x)
    
    def get_config(self):
        return {'_lambda': self._lambda, 'eta': self.eta, 'gamma': self.gamma, 
                'tau': self.tau, 'num_filters': self.num_filters}
    
    def rc_loss(self, weights):

        # flatten weights
        flattened = tf.reshape(weights, [-1, self.num_filters])

        # mask out filters whose summed absolute values < theta
        mask = tf.subtract(tf.reduce_mean(tf.abs(flattened), axis=[0]), self.tau) > 0
        num_unmasked = tf.reduce_sum(tf.cast(mask, tf.float32))
        masked = tf.boolean_mask(flattened, mask, axis=1)

        # create pearson coefficient matrix between unmasked filters
        means = tf.reduce_mean(masked, axis=[0], keepdims=True)
        diff = masked - means
        diff_norm = tf.nn.l2_normalize(diff, axis=[0])
        pearson = tf.tensordot(tf.transpose(diff_norm), diff_norm, axes=1)

        # substract identity matrix and get L-2 norm
        R_c = tf.norm(tf.subtract(pearson, tf.linalg.diag_part(pearson) * tf.eye(num_unmasked)))

        # divide by number of unmasked filters
        return R_c / num_unmasked
    
    def rs_loss(self, weights):
        flattened = tf.reshape(weights, [-1, self.num_filters, self.num_filters])

        channel_wise =  tf.reduce_sum(tf.norm(flattened, axis=[0, 1]))
        filter_wise =  tf.reduce_sum(tf.norm(flattened, axis=[0, 2]))

        return (channel_wise + filter_wise) / 2

# masks the kernel to remove filters who's mean absolute sum < tau
# only masks during testing phase
class SparseConv2D(layers.Conv2D):
    def __init__(self, filters, kernel_size, tau=1e-4, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.mask = tf.ones(filters)
        self.tau = tau
        self.last_phase = 1

    def call(self, inputs):        
        if tf.keras.backend.learning_phase(): # if training
            self.last_phase = 1    
            return super().call(inputs)
        
        else: # if testing phase
            if self.last_phase: # if mask hasn't been updated
                self.set_mask()
                self.last_phase = 0
            
            return tf.keras.backend.conv2d(inputs,
                                           self.kernel * tf.reshape(mask, [1, 1, 1, self.filters]),
                                           strides=self.strides,
                                           padding=self.padding,
                                           data_format=self.data_format,
                                           dilation_rate=self.dilation_rate)
    def set_mask(self):
        flattened = tf.reshape(self.kernel, [-1, self.filters])
        bool_mask = tf.subtract(tf.reduce_mean(tf.abs(flattened), axis=[0]), self.tau) > 0
        self.mask = tf.cast(bool_mask, tf.float32)
