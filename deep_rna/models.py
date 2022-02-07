import tensorflow as tf
import tensorflow.keras.layers as L
from deep_rna.spektral.layers import GraphMasking, SimpleGCN
    
class RNABodySmallModel(tf.keras.Model):
    '''Input: Spektral's BatchLoader of [node_features, edge_features] 
              there is a mask attached in node_features[:,:,-1:]
       Output: Embeded Node Features of shape (batch, seq_len, hidden_dim)
    '''
    def __init__(self, n_labels=5, hidden_dim=128, n_layers=2):
        super().__init__()
        
        self.graphmask = GraphMasking()
        self.pre_dense = L.Dense(hidden_dim,activation='linear',use_bias=False)
        self.gru = L.Bidirectional(L.GRU(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal'))
        self.lstm = L.Bidirectional(L.LSTM(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal'))
        self.gnn_layers = [SimpleGCN(hidden_dim) for _ in range(n_layers)]
        self.mask = None
    
    def call(self, inputs):
        node_inputs, edge_inputs = inputs
        
        edge_inputs = edge_inputs[..., 0] # use only the first edge feature at the moment
        node_inputs = self.graphmask(node_inputs)
        node_embed = self.pre_dense(node_inputs)

        node_embed = self.gru(node_embed)
        node_extra = self.gnn_layers[0]([node_embed, edge_inputs])
        node_embed = L.Concatenate()([node_embed, node_extra])

        
        node_embed = self.lstm(node_embed)
        node_extra = self.gnn_layers[1]([node_embed, edge_inputs])
        node_embed = L.Concatenate()([node_embed, node_extra])
        return node_embed
    
class RNAPredictionModel(tf.keras.Model):
    def __init__(self, body_model, n_labels=5):
        super().__init__()
        
        self.body_model = body_model
        self.final_dense = L.Dense(n_labels, 'linear')
        self.mask = None
        
    def dynamic_masked_mcrmse(self,y_true, y_pred):
        
        # self.mask needs to be dynamically updated for each batch
        # here, we provide two possible losses
        def mcrmse(y_true, y_pred):
            loss_square = tf.square(y_true - y_pred)
            if self.mask is not None:
                mask = tf.cast(self.mask,tf.float32)
                loss_square *= tf.expand_dims(mask,axis=-1)
            colwise_mse = tf.reduce_mean(loss_square, axis=(0, 1))
            
            mask_shape = tf.shape(mask)
            padded_total = tf.cast(mask_shape[0]*mask_shape[1], tf.float32)
            normalized = padded_total/tf.math.reduce_sum(mask)
            
            # counter-effect the effect of padded-zero making loss function too small
            return tf.reduce_mean(tf.sqrt(colwise_mse), axis=-1)*normalized

        return mcrmse(y_true, y_pred)
    
    def dynamic_masked_mse(self,y_true, y_pred):
        def mse(y_true, y_pred):
            loss_square = tf.square(y_true - y_pred)
            if self.mask is not None:
                mask = tf.cast(self.mask,tf.float32)
                loss_square *= tf.expand_dims(mask,axis=-1)
            mse = tf.reduce_mean(loss_square)
            
            mask_shape = tf.shape(mask)
            padded_total = tf.cast(mask_shape[0]*mask_shape[1], tf.float32)
            normalized = padded_total/tf.math.reduce_sum(mask)

            return mse*normalized

        return mse(y_true, y_pred)
    
    def call(self, inputs):
        node_embed = self.body_model(inputs)
        out = self.final_dense(node_embed)
        return out
    
    @tf.function
    def train_step(self, data):
        
        x, y = data
        self.mask = self.body_model.graphmask.compute_mask(x[0])

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):
        
        x, y = data
        self.mask = self.body_model.graphmask.compute_mask(x[0])
        y_pred = self(x, training=False)  # Forward pass

        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
