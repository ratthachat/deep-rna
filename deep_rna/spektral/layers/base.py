import numpy as np
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import smart_cond

from spektral.layers import ops


class Disjoint2Batch(Layer):
    r"""Utility layer that converts data from disjoint mode to batch mode by
    zero-padding the node features and adjacency matrices.

    **Mode**: disjoint.

    **This layer expects a sparse adjacency matrix.**

    **Input**

    - Node features of shape `(n_nodes, n_node_features)`;
    - Binary adjacency matrix of shape `(n_nodes, n_nodes)`;
    - Graph IDs of shape `(n_nodes, )`;

    **Output**

    - Batched node features of shape `(batch, N_max, n_node_features)`;
    - Batched adjacency matrix of shape `(batch, N_max, N_max)`;
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs, **kwargs):
        X, A, I = inputs

        batch_X = ops.disjoint_signal_to_batch(X, I)
        batch_A = ops.disjoint_adjacency_to_batch(A, I)

        # Ensure that the channel dimension is known
        batch_X.set_shape((None, None, X.shape[-1]))
        batch_A.set_shape((None, None, None))

        return batch_X, batch_A


class GraphMasking(Layer):
    """
    A layer that starts the propagation of masks in a model.

    This layer assumes that the node features given as input have been extended with a
    binary mask that indicates which nodes are valid in each graph.
    The layer is useful when using a `data.BatchLoader` with `mask=True` or in general
    when zero-padding graphs so that all batches have the same size. The binary mask
    indicates with a 1 those nodes that should be taken into account by the model.

    The layer will remove the rightmost feature from the nodes and start a mask
    propagation to all subsequent layers:

    ```python
    print(x.shape)  # shape (batch, n_nodes, n_node_features + 1)
    mask = x[..., -1:]  # shape (batch, n_nodes, 1)
    x_new = x[..., :-1] # shape (batch, n_nodes, n_node_features)
    ```

    """

    def compute_mask(self, inputs, mask=None):
        x = inputs[0] if isinstance(inputs, list) else inputs
        mask = tf.cast(x[..., -1:],tf.bool)
        mask = tf.squeeze(mask)
        mask = tf.ensure_shape(mask, [None, None])
        return mask

    def call(self, inputs, **kwargs):
        # Remove mask from features
        if isinstance(inputs, list):
            inputs[0] = inputs[0][..., :-1]
        else:
            inputs = inputs[..., :-1]

        return inputs


class InnerProduct(Layer):
    r"""
    Computes the inner product between elements of a 2d Tensor:
    $$
        \langle \x, \x \rangle = \x\x^\top.
    $$

    **Mode**: single.

    **Input**

    - Tensor of shape `(n_nodes, n_features)`;

    **Output**

    - Tensor of shape `(n_nodes, n_nodes)`.

    :param trainable_kernel: add a trainable square matrix between the inner
    product (e.g., `X @ W @ X.T`);
    :param activation: activation function;
    :param kernel_initializer: initializer for the weights;
    :param kernel_regularizer: regularization applied to the kernel;
    :param kernel_constraint: constraint applied to the kernel;
    """

    def __init__(
        self,
        trainable_kernel=False,
        activation=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.trainable_kernel = trainable_kernel
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        if self.trainable_kernel:
            features_dim = input_shape[-1]
            self.kernel = self.add_weight(
                shape=(features_dim, features_dim),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        self.built = True

    def call(self, inputs):
        if self.trainable_kernel:
            output = K.dot(K.dot(inputs, self.kernel), K.transpose(inputs))
        else:
            output = K.dot(inputs, K.transpose(inputs))
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {
            "trainable_kernel": self.trainable_kernel,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MinkowskiProduct(Layer):
    r"""
    Computes the hyperbolic inner product between elements of a rank 2 Tensor:
    $$
        \langle \x, \x \rangle = \x \,
        \begin{pmatrix}
            \I_{d \times d} & 0 \\
            0              & -1
        \end{pmatrix} \, \x^\top.
    $$

    **Mode**: single.

    **Input**

    - Tensor of shape `(n_nodes, n_features)`;

    **Output**

    - Tensor of shape `(n_nodes, n_nodes)`.

    :param activation: activation function;
    """

    def __init__(self, activation=None, **kwargs):

        super().__init__(**kwargs)
        self.activation = activations.get(activation)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.built = True

    def call(self, inputs):
        F = tf.shape(inputs)[-1]
        minkowski_prod_mat = np.eye(F)
        minkowski_prod_mat[-1, -1] = -1.0
        minkowski_prod_mat = K.constant(minkowski_prod_mat)
        output = K.dot(inputs, minkowski_prod_mat)
        output = K.dot(output, K.transpose(inputs))
        output = K.clip(output, -10e9, -1.0)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {"activation": self.activation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SparseDropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    Arguments:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    noise_shape: 1D integer tensor representing the shape of the
      binary dropout mask that will be multiplied with the input.
      For instance, if your inputs have shape
      `(batch_size, timesteps, features)` and
      you want the dropout mask to be the same for all timesteps,
      you can use `noise_shape=(batch_size, 1, features)`.
    seed: A Python integer to use as random seed.

    Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
    """

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    @staticmethod
    def _get_noise_shape(inputs):
        return tf.shape(inputs.values)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            return self.sparse_dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate,
            )

        output = smart_cond.smart_cond(training, dropped_inputs, lambda: inputs)
        return output

    def get_config(self):
        config = {"rate": self.rate, "noise_shape": self.noise_shape, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def sparse_dropout(x, rate, noise_shape=None, seed=None):
        random_tensor = tf.random.uniform(noise_shape, seed=seed, dtype=x.dtype)
        keep_prob = 1 - rate
        scale = 1 / keep_prob
        keep_mask = random_tensor >= rate
        output = tf.sparse.retain(x, keep_mask)
        # output = output * scale  # gradient issues with automatic broadcasting
        output = output * tf.reshape(
            tf.convert_to_tensor(scale, dtype=output.dtype), (1,) * output.shape.ndims
        )
        return output
