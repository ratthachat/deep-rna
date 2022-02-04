import numpy as np
import tensorflow as tf

def pad_jagged_array(x, target_shape):
    """
    Given a jagged array of arbitrary dimensions, zero-pads all elements in the
    array to match the provided `target_shape`.
    :param x: a list or np.array of dtype object, containing np.arrays with
    variable dimensions;
    :param target_shape: a tuple or list s.t. target_shape[i] >= x.shape[i]
    for each x in X. If `target_shape[i] = -1`, it will be automatically
    converted to X.shape[i], so that passing a target shape of e.g. (-1, n, m)
    will leave the first  dimension of each element untouched.
    :return: a np.array of shape `(len(x), ) + target_shape`.
    """
    if len(x) < 1:
        raise ValueError("Jagged array cannot be empty")
    target_len = len(x)
    target_shape = tuple(
        shp if shp != -1 else x[0].shape[j] for j, shp in enumerate(target_shape)
    )
    output = np.zeros((target_len,) + target_shape, dtype=x[0].dtype)
    for i in range(target_len):
        slc = (i,) + tuple(slice(shp) for shp in x[i].shape)
        output[slc] = x[i]

    return output

def to_batch(x_list=None, a_list=None, e_list=None, mask=False):
    """
    Converts lists of node features, adjacency matrices and edge features to
    [batch mode](https://graphneural.network/data-modes/#batch-mode),
    by zero-padding all tensors to have the same node dimension `n_max`.

    Either the node features or the adjacency matrices must be provided as input.

    The i-th element of each list must be associated with the i-th graph.

    If `a_list` contains sparse matrices, they will be converted to dense
    np.arrays.

    The edge attributes of a graph can be represented as

    - a dense array of shape `(n_nodes, n_nodes, n_edge_features)`;
    - a sparse edge list of shape `(n_edges, n_edge_features)`;

    and they will always be returned as dense arrays.

    :param x_list: a list of np.arrays of shape `(n_nodes, n_node_features)`
    -- note that `n_nodes` can change between graphs;
    :param a_list: a list of np.arrays or scipy.sparse matrices of shape
    `(n_nodes, n_nodes)`;
    :param e_list: a list of np.arrays of shape
    `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`;
    :param mask: bool, if True, node attributes will be extended with a binary mask that
    indicates valid nodes (the last feature of each node will be 1 if the node is valid
    and 0 otherwise). Use this flag in conjunction with layers.base.GraphMasking to
    start the propagation of masks in a model.

    :return: only if the corresponding list is given as input:

        -  `x`: np.array of shape `(batch, n_max, n_node_features)`;
        -  `a`: np.array of shape `(batch, n_max, n_max)`;
        -  `e`: np.array of shape `(batch, n_max, n_max, n_edge_features)`;
    """
    if a_list is None and x_list is None:
        raise ValueError("Need at least x_list or a_list")

    n_max = max([x.shape[0] for x in (x_list if x_list is not None else a_list)])

    # Node features
    x_out = None
    if x_list is not None:
        if mask:
            x_list = [np.concatenate((x, np.ones((x.shape[0], 1))), -1) for x in x_list]
        x_out = pad_jagged_array(x_list, (n_max, -1))

    # Adjacency matrix
    a_out = None
    if a_list is not None:
        if hasattr(a_list[0], "toarray"):  # Convert sparse to dense
            a_list = [a.toarray() for a in a_list]
        a_out = pad_jagged_array(a_list, (n_max, n_max))

    # Edge attributes
    e_out = None
    if e_list is not None:
        if e_list[0].ndim == 2:  # Sparse to dense
            for i in range(len(a_list)):
                a, e = a_list[i], e_list[i]
                e_new = np.zeros(a.shape + e.shape[-1:])
                e_new[np.nonzero(a)] = e
                e_list[i] = e_new
        e_out = pad_jagged_array(e_list, (n_max, n_max, -1))

    return tuple(out for out in [x_out, a_out, e_out] if out is not None)

def batch_generator(data, batch_size=32, epochs=None, shuffle=True):
    """
    Iterates over the data for the given number of epochs, yielding batches of
    size `batch_size`.
    :param data: np.array or list of np.arrays with the same first dimension;
    :param batch_size: number of samples in a batch;
    :param epochs: number of times to iterate over the data (default None, iterates
    indefinitely);
    :param shuffle: whether to shuffle the data at the beginning of each epoch
    :return: batches of size `batch_size`.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    if len(data) < 1:
        raise ValueError("data cannot be empty")
    if len({len(item) for item in data}) > 1:
        raise ValueError("All inputs must have the same __len__")

    if epochs is None or epochs == -1:
        epochs = np.inf
    len_data = len(data[0])
    batches_per_epoch = int(np.ceil(len_data / batch_size))
    epoch = 0
    while epoch < epochs:
        epoch += 1
        if shuffle:
            shuffle_inplace(*data)
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)
            to_yield = [item[start:stop] for item in data]
            if len(data) == 1:
                to_yield = to_yield[0]

            yield to_yield


def shuffle_inplace(*args):
    rng_state = np.random.get_state()
    for a in args:
        np.random.set_state(rng_state)
        np.random.shuffle(a)


def get_spec(x):
    if isinstance(x, tf.SparseTensor):
        return tf.SparseTensorSpec
    else:
        return tf.TensorSpec


def prepend_none(t):
    return (None,) + t


def to_tf_signature(signature):
    """
    Converts a Dataset signature to a TensorFlow signature.
    :param signature: a Dataset signature.
    :return: a TensorFlow signature.
    """
    output = []
    keys = ["x", "a", "e", "i"]
    for k in keys:
        if k in signature:
            shape = signature[k]["shape"]
            dtype = signature[k]["dtype"]
            spec = signature[k]["spec"]
            output.append(spec(shape, dtype))
    output = tuple(output)
    if "y" in signature:
        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]
        output = (output, spec(shape, dtype))

    return output
