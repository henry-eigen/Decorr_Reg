# Decorr_Reg

Implementation of [Improving Deep Neural Network Sparsity through Decorrelation Regularization](https://www.ijcai.org/proceedings/2018/0453.pdf)

It combined a regularization penalty with activation inhibition to encourages convolutional layers to learn a sparse, diverse set of kernels.

Can be used with keras convolutional layer like:

```
from ssr import SparseConv2D, rc_reg

x = SparseConv2D(...
            ...
            ...
            kernel_regularizer = rc_reg(num_channels))(x)
```

The ```num_channels``` parameter should specify the number of channels in the previous layer's output. Note that this has nothing to do with the behavior of the regularization itself and everything to do with the fact that Tensorflow decided it couldn't guess the shape itself. 
