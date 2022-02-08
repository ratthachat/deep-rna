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

class Conv1dBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, strides=1,hidden_dim=128):
        super(Conv1dBlock, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.hidden_dim = hidden_dim
        
#         self.conv_layer = L.DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding='same')
        self.conv_layer = L.Conv1D(hidden_dim, kernel_size=kernel_size, strides=strides, padding='same')
        self.norm_layer = L.LayerNormalization()
#         self.inverted_layer = L.Dense(hidden_dim*1)
        self.dense_layer = L.Dense(hidden_dim)
        self.add_layer = L.Concatenate()#L.Add()
    def call(self, node_feat_input):
        node_feat = self.conv_layer(node_feat_input) # tf >= 2.7.0
        node_feat = self.norm_layer(node_feat)
#         node_feat = self.inverted_layer(node_feat)
        node_feat = tf.keras.activations.gelu(node_feat, approximate=True)
        node_feat = self.dense_layer(node_feat)
        node_feat = self.add_layer([node_feat, node_feat_input])
        return node_feat
    
class RNABodyDeepModel(tf.keras.Model):
    '''Input: Spektral's BatchLoader of [node_features, edge_features] 
              there is a mask attached in node_features[:,:,-1:]
       Output: Embeded Node Features of shape (batch, seq_len, hidden_dim)
    '''
    def __init__(self, n_labels=5, hidden_dim=128, n_conv_layers=2, n_gnn_layers=2, n_edge_features=3):
        super().__init__()

        self.graphmask = GraphMasking()
        self.hidden_dim = hidden_dim
        self.n_edge_features = n_edge_features
        self.n_conv_layers = n_conv_layers
        self.n_gnn_layers = n_gnn_layers
        self.pre_dense = L.Dense(hidden_dim,activation='linear',use_bias=False)
        self.rnn_layers = [L.Bidirectional(L.GRU(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal')),
                           L.Bidirectional(L.LSTM(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal'))]
        self.gnn_layers = [[SimpleGCN(hidden_dim//2) for _ in range(n_edge_features)] for _ in range(n_gnn_layers)]
        self.conv_blocks = [Conv1dBlock() for _ in range(n_conv_layers)]
        self.mask = None
    
    def call(self, inputs):
        node_inputs, edge_inputs = inputs

        node_inputs = self.graphmask(node_inputs)
        node_embed = self.pre_dense(node_inputs)
        
        for i in range(self.n_conv_layers):
            node_embed = self.conv_blocks[i](node_embed)
        
            node_embed0 = node_embed
            for k in range(self.n_edge_features):
                edge_inputs0 = edge_inputs[..., k]
                node_extra = self.gnn_layers[i][k]([node_embed0, edge_inputs0])
                node_embed = L.Concatenate()([node_embed, node_extra])
                
            node_embed = self.rnn_layers[i](node_embed)
        return node_embed

class RNAPredictionModel(tf.keras.Model):
    def __init__(self, body_model, n_labels=5, activation='linear'):
        super().__init__()

        self.body_model = body_model
        self.final_dense = L.Dense(n_labels, activation)
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

def RNAPretrainedModel(
                 model_size='small',
                 include_top=True, 
                 weights='openvaccine', 
                 n_labels=5,
                 activation='linear'):

    small_model_url = "https://drive.google.com/u/0/uc?id=1Yyc_143ZQeTaCVcTCDv-WQjw6NZ8j0FT&export=download"

    assert model_size == 'small' or model_size == 'deep'
    assert n_labels > 1

    if model_size == 'small':
        model_body = RNABodySmallModel()
        input_shape = [(None, None, 14), (None, None, None, None)] # fix for the pretrained model

    if weights is not None:
        model_prediction = RNAPredictionModel(model_body, n_labels=5, activation=activation)
        model_prediction.build(input_shape = input_shape)

        file_path = tf.keras.utils.get_file(fname='model.h5',origin=small_model_url)
        model_prediction.load_weights(file_path)
        print('load pretrained open-vaccine weights successfully....!')

        if n_labels != 5: # change head, but with pretrained body
            print('Note that since the n_labels is different from OpenVaccine classes, the prediction dense will be fresh with random weights.')
            model_prediction = RNAPredictionModel(model_body, n_labels=n_labels, activation=activation)
            model_prediction.build(input_shape = input_shape)
    else:
        model_prediction = RNAPredictionModel(model_body, n_labels=n_labels, activation=activation)
        model_prediction.build(input_shape = input_shape)

    if include_top == False:
        return model_body

    return model_prediction
