import numpy as np
import tensorflow as tf

from deep_rna.spektral.data.utils import (
    batch_generator,
    get_spec,
    prepend_none,
    to_batch,
    to_tf_signature,
    pad_jagged_array,
)

version = tf.__version__.split(".")
major, minor = int(version[0]), int(version[1])
tf_loader_available = major >= 2 and minor >= 4


class Loader:
    """
    Parent class for data loaders. The role of a Loader is to iterate over a
    Dataset and yield batches of graphs to feed your Keras Models.

    This is achieved by having a generator object that produces lists of Graphs,
    which are then collated together and returned as Tensors.

    The core of a Loader is the `collate(batch)` method.
    This takes as input a list of `Graph` objects and returns a list of Tensors,
    np.arrays, or SparseTensors.

    For instance, if all graphs have the same number of nodes and size of the
    attributes, a simple collation function can be:

    ```python
    def collate(self, batch):
        x = np.array([g.x for g in batch])
        a = np.array([g.a for g in batch)]
        return x, a
    ```

    The `load()` method of a Loader returns an object that can be passed to a Keras
    model when using the `fit`, `predict` and `evaluate` functions.
    You can use it as follows:

    ```python
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch)
    ```

    The `steps_per_epoch` property represents the number of batches that are in
    an epoch, and is a required keyword when calling `fit`, `predict` or `evaluate`
    with a Loader.

    If you are using a custom training function, you can specify the input signature
    of your batches with the tf.TypeSpec system to avoid unnecessary re-tracings.
    The signature is computed automatically by calling `loader.tf_signature()`.

    For example, a simple training step can be written as:

    ```python
    @tf.function(input_signature=loader.tf_signature())  # Specify signature here
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ```

    We can then train our model in a loop as follows:

    ```python
    for batch in loader:
        train_step(*batch)
    ```

    **Arguments**

    - `dataset`: a `spektral.data.Dataset` object;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the dataset at the start of each epoch.
    """

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self):
        """
        Returns lists (batches) of `Graph` objects.
        """
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )

    def collate(self, batch):
        """
        Converts a list of graph objects to Tensors or np.arrays representing the batch.
        :param batch: a list of `Graph` objects.
        """
        raise NotImplementedError

    def load(self):
        """
        Returns an object that can be passed to a Keras model when using the `fit`,
        `predict` and `evaluate` functions.
        By default, returns the Loader itself, which is a generator.
        """
        return self

    def tf_signature(self):
        """
        Returns the signature of the collated batches using the tf.TypeSpec system.
        By default, the signature is that of the dataset (`dataset.signature`):

            - Adjacency matrix has shape `[n_nodes, n_nodes]`
            - Node features have shape `[n_nodes, n_node_features]`
            - Edge features have shape `[n_edges, n_node_features]`
            - Targets have shape `[..., n_labels]`
        """
        signature = self.dataset.signature
        return to_tf_signature(signature)

    def pack(self, batch):
        """
        Given a batch of graphs, groups their attributes into separate lists and packs
        them in a dictionary.

        For instance, if a batch has three graphs g1, g2 and g3 with node
        features (x1, x2, x3) and adjacency matrices (a1, a2, a3), this method
        will return a dictionary:

        ```python
        >>> {'a_list': [a1, a2, a3], 'x_list': [x1, x2, x3]}
        ```

        :param batch: a list of `Graph` objects.
        """
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.dataset.signature.keys()]
        return dict(zip(keys, output))

    @property
    def steps_per_epoch(self):
        """
        :return: the number of batches of size `self.batch_size` in the dataset (i.e.,
        how many batches are in an epoch).
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))

class BatchLoader(Loader):
    """
    A Loader for [batch mode](https://graphneural.network/data-modes/#batch-mode).

    This loader returns batches of graphs stacked along an extra dimension,
    with all "node" dimensions padded to be equal among all graphs.

    If `n_max` is the number of nodes of the biggest graph in the batch, then
    the padding consist of adding zeros to the node features, adjacency matrix,
    and edge attributes of each graph so that they have shapes
    `(n_max, n_node_features)`, `(n_max, n_max)`, and
    `(n_max, n_max, n_edge_features)` respectively.

    The zero-padding is done batch-wise, which saves up memory at the cost of
    more computation. If latency is an issue but memory isn't, or if the
    dataset has graphs with a similar number of nodes, you can use
    the `PackedBatchLoader` that first zero-pads all the dataset and then
    iterates over it.

    Note that the adjacency matrix and edge attributes are returned as dense
    arrays (mostly due to the lack of support for sparse tensor operations for
    rank >2).

    Only graph-level labels are supported with this loader (i.e., labels are not
    zero-padded because they are assumed to have no "node" dimensions).

    **Arguments**

    - `dataset`: a graph Dataset;
    - `mask`: if True, node attributes will be extended with a binary mask that
    indicates valid nodes (the last feature of each node will be 1 if the node is valid
    and 0 otherwise). Use this flag in conjunction with layers.base.GraphMasking to
    start the propagation of masks in a model.
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[batch, n_max, n_node_features]`;
    - `a`: adjacency matrices of shape `[batch, n_max, n_max]`;
    - `e`: edge attributes of shape `[batch, n_max, n_max, n_edge_features]`.

    `labels` have shape `[batch, n_labels]`.
    """

    def __init__(self, dataset, mask=False, batch_size=1, epochs=None, shuffle=True):
        self.mask = mask
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            if self.mask: # padding apply
                n_max = max([yy.shape[0] for yy in y])
                y = pad_jagged_array(y, (n_max, -1))

            y = np.array(y)

        output = to_batch(**packed, mask=self.mask)
        

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def tf_signature(self):
        """
        Adjacency matrix has shape [batch, n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_nodes, n_nodes, n_edge_features]
        Targets have shape [batch, ..., n_labels]
        """
        signature = self.dataset.signature
        for k in signature:
            signature[k]["shape"] = prepend_none(signature[k]["shape"])
        if "x" in signature:
            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )
        if "a" in signature:
            # Adjacency matrix in batch mode is dense
            signature["a"]["spec"] = tf.TensorSpec
        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature["e"]["shape"] = prepend_none(signature["e"]["shape"])

        return to_tf_signature(signature)


class PackedBatchLoader(BatchLoader):
    """
    A `BatchLoader` that zero-pads the graphs before iterating over the dataset.
    This means that `n_max` is computed over the whole dataset and not just
    a single batch.

    While using more memory than `BatchLoader`, this loader should reduce the
    computational overhead of padding each batch independently.

    Use this loader if:

    - memory usage isn't an issue and you want to produce the batches as fast
    as possible;
    - the graphs in the dataset have similar sizes and there are no outliers in
    the dataset (i.e., anomalous graphs with many more nodes than the dataset
    average).

    **Arguments**

    - `dataset`: a graph Dataset;
    - `batch_size`: size of the mini-batches;
    - `epochs`: number of epochs to iterate over the dataset. By default (`None`)
    iterates indefinitely;
    - `shuffle`: whether to shuffle the data at the start of each epoch.

    **Output**

    For each batch, returns a tuple `(inputs, labels)`.

    `inputs` is a tuple containing:

    - `x`: node attributes of shape `[batch, n_max, n_node_features]`;
    - `a`: adjacency matrices of shape `[batch, n_max, n_max]`;
    - `e`: edge attributes of shape `[batch, n_max, n_max, n_edge_features]`.

    `labels` have shape `[batch, ..., n_labels]`.
    """

    def __init__(self, dataset, mask=False, batch_size=1, epochs=None, shuffle=True):
        super().__init__(
            dataset, mask=mask, batch_size=batch_size, epochs=epochs, shuffle=shuffle
        )

        # Drop the Dataset container and work on packed tensors directly
        packed = self.pack(self.dataset)

        y = packed.pop("y_list", None)
        if y is not None:
            y = np.array(y)

        self.signature = dataset.signature
        self.dataset = to_batch(**packed, mask=mask)
        if y is not None:
            self.dataset += (y,)

        # Re-instantiate generator after packing dataset
        self._generator = self.generator()

    def collate(self, batch):
        if len(batch) == 2:
            # If there is only one input, i.e., batch = [x, y], we unpack it
            # like this because Keras does not support input lists with only
            # one tensor.
            return batch[0], batch[1]
        else:
            return batch[:-1], batch[-1]

    def tf_signature(self):
        """
        Adjacency matrix has shape [batch, n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_nodes, n_nodes, n_edge_features]
        Targets have shape [batch, ..., n_labels]
        """
        signature = self.signature
        for k in signature:
            signature[k]["shape"] = prepend_none(signature[k]["shape"])
        if "x" in signature:
            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )
        if "a" in signature:
            # Adjacency matrix in batch mode is dense
            signature["a"]["spec"] = tf.TensorSpec
        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature["e"]["shape"] = prepend_none(signature["e"]["shape"])

        return to_tf_signature(signature)

    @property
    def steps_per_epoch(self):
        if len(self.dataset) > 0:
            return int(np.ceil(len(self.dataset[0]) / self.batch_size))
