import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

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
