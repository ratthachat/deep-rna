import tensorflow as tf
import tensorflow.keras.layers as L
from deep_rna.spektral.layers import GraphMasking, SimpleGCN

class RNABodySmallModel(tf.keras.Model):
    '''A small baseline RNA body model of 700K parameters resulted in decent RNA embedding'''
    def __init__(self, n_labels=5, hidden_dim=128, n_layers=2):
        super().__init__()

        self.graphmask = GraphMasking()
        self.pre_dense = L.Dense(hidden_dim,activation='linear',use_bias=False)
        self.gru = L.Bidirectional(L.GRU(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal'))
        self.lstm = L.Bidirectional(L.LSTM(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal'))
        self.gnn_layers = [SimpleGCN(hidden_dim) for _ in range(n_layers)]
        self.mask = None

    def call(self, inputs):
        '''Input: Spektral's BatchLoader of [node_features, edge_features] 
              there is a mask attached in node_features[:,:,-1:]
           Output: Embeded Node Features of shape (batch, seq_len, hidden_dim)
        '''

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
    '''Simplified OpenVaccine SOTA Recipe of Conv1d Block'''
    def __init__(self, kernel_size=3, strides=1,hidden_dim=128,drop_rate=0.25):
        super(Conv1dBlock, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.hidden_dim = hidden_dim
        self.conv_layer = L.Conv1D(hidden_dim, kernel_size=kernel_size, strides=strides, padding='same')
        self.norm_layer1 = L.LayerNormalization()
        self.inverted_layer = L.Dense(hidden_dim*3)
        self.dense_layer = L.Dense(hidden_dim)
        self.drop_layer = L.Dropout(drop_rate, noise_shape=[None,1,1])
        self.add_layer = L.Concatenate()#L.Add()
        self.norm_layer2 = L.LayerNormalization()
    def call(self, node_feat_input):
        node_feat = self.conv_layer(node_feat_input) # tf >= 2.7.0
        node_feat = self.norm_layer1(node_feat)
        node_feat = self.inverted_layer(node_feat)
        node_feat = tf.keras.activations.gelu(node_feat, approximate=True)
        node_feat = self.dense_layer(node_feat)
        node_feat = self.norm_layer2(node_feat)
        node_feat = self.drop_layer(node_feat)
        node_feat = self.add_layer([node_feat, node_feat_input])
        return node_feat
    
class RNABodyDeepModel(tf.keras.Model):
    '''A deep RNA body model of 6.8M parameters resulted in SOTA RNA embedding
    This body model simplifies SOTA (ie. gold-solution) model used in OpenVaccine competition
    i.e. https://www.kaggle.com/group16/covid-19-mrna-4th-place-solution
    but still provide a competitive embeding quality.
    '''
    def __init__(self, n_labels=5, hidden_dim=128, n_layers=2, n_edge_features=5):
        super().__init__()

        self.graphmask = GraphMasking()
        self.hidden_dim = hidden_dim
        self.n_edge_features = n_edge_features+1 # plus learnable edge feat
        self.n_layers = n_layers
        self.edge_dense = L.Dense(1,activation='relu',use_bias=False)
        self.pre_dense = L.Dense(hidden_dim,activation='linear',use_bias=False)
        self.rnn_layers = [L.Bidirectional(L.GRU(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal')),
                           L.Bidirectional(L.LSTM(hidden_dim, dropout=0.25, return_sequences=True, kernel_initializer='orthogonal'))]
        self.gnn_layers = [[SimpleGCN(hidden_dim) for _ in range(self.n_edge_features)] for _ in range(n_layers)]
        self.conv_blocks = [Conv1dBlock() for _ in range(n_layers)]
        self.concat = L.Concatenate()
        self.mask = None
        
        # Simplified OpenVaccine SOTA Recipe
        self.conv_block6 = Conv1dBlock(kernel_size=6, hidden_dim=64,drop_rate=0.0)
        self.conv_block15 = Conv1dBlock(kernel_size=15, hidden_dim=32,drop_rate=0.0)
        self.conv_block30 = Conv1dBlock(kernel_size=30, hidden_dim=16,drop_rate=0.0)
        self.ffn = [L.Dense(hidden_dim) for  _ in range(n_layers)]
        self.attention = L.MultiHeadAttention(num_heads=2, key_dim=32)
        self.norm_layer1 = [L.LayerNormalization() for  _ in range(self.n_edge_features)]
        self.norm_layer2 = L.LayerNormalization()
        
    def call(self, inputs):
        '''
        Input: Spektral's BatchLoader of [node_features, edge_features] 
              there is a mask attached in node_features[:,:,-1:]
        Output: Embeded Node Features of shape (batch, seq_len, hidden_dim)
        '''
        node_inputs, edge_inputs = inputs

        node_inputs = self.graphmask(node_inputs)
        node_embed = self.pre_dense(node_inputs)
        
        node_embed_list = [node_embed]
        node_embed_list.append(self.conv_block6(node_embed_list[-1]))
        node_embed_list.append(self.conv_block15(node_embed_list[-1]))
        node_embed_list.append(self.conv_block30(node_embed_list[-1]))
        
        node_embed = self.concat(node_embed_list)
        node_embed_saved = node_embed
        
        learned_edge = self.edge_dense(edge_inputs)
        edge_inputs = self.concat([edge_inputs, learned_edge])
        
        for i in range(self.n_layers):
            node_embed = self.conv_blocks[i](node_embed)
        
            node_embed0 = node_embed
            for k in range(self.n_edge_features):
                edge_inputs0 = edge_inputs[..., k]
                node_extra = self.gnn_layers[i][k]([node_embed0, edge_inputs0])
                node_extra = self.norm_layer1[k](node_extra)
                node_embed = self.concat([node_embed, node_extra])
            
            node_embed0 = self.concat([node_embed, node_embed_saved])
            
            node_embed = self.ffn[i](node_embed0)
            node_embed = self.norm_layer2(node_embed)
            node_embed = self.attention(node_embed, node_embed)
            
            node_embed = self.concat([node_embed, node_embed0])
            node_embed = self.rnn_layers[i](node_embed)
        return node_embed
    
class RNAPredictionModel(tf.keras.Model):
    ''' Prediction "head" model which support both "small" and "deep" bodies i.e.
    RNABodySmallModel and RNABodyDeepModel respectively. 
    This class define "dynamic masked loss" e.g. self.dynamic_masked_mcrmse,
    to support a batch consisting of RNAs of various length. 
    (all RNAs will be padded to have the same length as the longest one in the batch)
    The dynamic masked loss  will ignore the loss of padded RNAs in each batch.
    
    Arguments:
    body_model : an instance of either RNABodySmallModel or RNABodyDeepModel class
    n_labels : number of labels to predict
    activation : activation function of the final prediction layer
    class_weight (optional) : a list of "n_labels" element corresponding to each label's weight.
                              If class_weight is None, equal weight will be used.
    '''
    def __init__(self, body_model, n_labels=5, activation='linear', class_weight = None):
        super().__init__()

        self.body_model = body_model
        self.final_dense = L.Dense(n_labels, activation)
        self.class_weight = class_weight
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
            if self.class_weight is not None:
                # assert len(colwise_mse) == len(self.class_weight)
                colwise_mse *= self.class_weight
            
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

    def call(self, x):
        self.mask = self.body_model.graphmask.compute_mask(x[0])
        node_embed = self.body_model(x)
        out = self.final_dense(node_embed)
        return out

    @tf.function
    def train_step(self, data):

        x, y = data
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
        y_pred = self(x, training=False)  # Forward pass

        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def RNAPretrainedModel(
                 model_size='deep',
                 include_top=True, 
                 weights='openvaccine', 
                 n_labels=5,
                 activation='linear',
                 class_weight=None):
    ''' A function to load pretrained deep model (6.8M parameters)
    Arguments are in standard keras format.
    '''
    
    assert model_size == 'small' or model_size == 'deep'
    assert n_labels > 1

    if model_size == 'small':
        model_body = RNABodySmallModel()
        model_url = "https://drive.google.com/u/0/uc?id=1Yyc_143ZQeTaCVcTCDv-WQjw6NZ8j0FT&export=download"
        n_node_features = 14 # speficiation of the pretrained model
        n_edge_features = 3
        file_path = tf.keras.utils.get_file(fname='model_small.h5',origin=model_url)
    else:
        print('loading deep model...')
        model_body = RNABodyDeepModel()
        model_url = "https://drive.google.com/u/0/uc?id=1zTRijMYq1maCWBr42kikFh3Qh9KBQVuE&export=download"
        n_node_features = 23 # speficiation of the pretrained model
        n_edge_features = 5
        file_path = tf.keras.utils.get_file(fname='model_deep.h5',origin=model_url)
        
    input_shape = [(None, None, n_node_features), (None, None, None,n_edge_features)] 
    if weights is not None:
        model_prediction = RNAPredictionModel(model_body, n_labels=5, activation=activation,class_weight=class_weight)
        model_prediction.build(input_shape = input_shape)

        if model_size == 'deep': # small model has no good pretrained weights
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
