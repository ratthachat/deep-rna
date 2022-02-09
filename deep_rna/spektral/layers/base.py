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


class SimpleGCN(tf.keras.layers.Layer):
    '''
    Simplest Unnormalized GCN Convolutional Layer.
    Easily use with other keras model's calling:
  
    # node_states, adjacency have to be explicitly given
    ...
    new_node_states = SimpleGCN(256)([node_states, adjacency])
    new_node_states = Dense(128)(new_node_states)
    ...
    
    '''
    def __init__(
        self,
        units=128,
        activation='linear',
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.supports_masking = True
        
        super().__init__(
            **kwargs
        )
        
    def build(self, input_shape):
        node_input_dim = input_shape[0][-1]
        self.kernel =  self.add_weight(
            shape=(node_input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
        )
        
        self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                name="bias",
        )
        
        self.built = True
    
    def call(self, inputs, training=False, mask=None):
        node_states, adjacency = inputs
        node_states_aggregated = tf.matmul(adjacency, node_states)
        node_states_aggregated = tf.matmul(node_states_aggregated, self.kernel)
        
        node_states_aggregated += self.bias
        if mask:
            node_states_aggregated = self._apply_mask(node_states_aggregated, mask)

        return self.activation(node_states_aggregated)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def _apply_mask(self, inputs, mask):
        if len(tf.shape(inputs)) > len(tf.shape(mask[0])):
            return inputs * tf.cast(tf.expand_dims(mask[0],axis=-1),inputs.dtype)
        else:
            return inputs * tf.cast(mask[0],inputs.dtype)
    
    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}
