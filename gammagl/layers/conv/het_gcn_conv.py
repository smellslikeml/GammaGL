import tensorlayerx as tlx
from gammagl.utils import degree
from gammagl.layers.conv import MessagePassing
from gammagl.mpops import *



class HetGCNConv(MessagePassing):
    r"""The heterophily-informed message passing graph convolutional
    operator (HIMP) from the `"Heterophily-Informed Message Passing:
    Enhancing Graph Neural Networks for Heterophilic Graphs"
    <https://arxiv.org/abs/2504.19785>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where each message sent along edge :math:`(j, i)` is additionally scaled
    by a parameter-free, similarity-derived weight :math:`s_{ij}` computed
    from the post-linear node features:

    .. math::
        s_{ij} = \begin{cases}
            1, & \text{if}~\texttt{het\_mode}=\texttt{'original'} \\
            \frac{\mathbf{x}_i^{\top} \mathbf{x}_j}
                 {\|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|},
              & \text{if}~\texttt{het\_mode}=\texttt{'homophily'} \\
            1 - \frac{\mathbf{x}_i^{\top} \mathbf{x}_j}
                 {\|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|},
              & \text{if}~\texttt{het\_mode}=\texttt{'heterophily'} \\
        \end{cases}

    so that the layer can emphasize homophilous or heterophilous edges.
    With the default :obj:`het_mode='original'` (:math:`s_{ij} = 1`), this
    layer reduces exactly to a plain
    :class:`gammagl.layers.conv.GCNConv`. It introduces no new learnable
    parameters.

    Parameters
    ----------
    in_channels: int
        Size of each input sample.
    out_channels: int
        Size of each output sample.
    norm: str, optional
        How to apply the normalizer.  Can be one of the following values:

            * ``right``, to divide the aggregated messages by each node's in-degrees, which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}` above, equivalent to symmetric normalization.

            * ``left``, to divide the messages sent out from each node by its out-degrees, equivalent to random walk normalization.
    add_bias: bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

    """

    def __init__(self,
                in_channels,
                out_channels,
                norm = 'both',
                add_bias = True):
        super().__init__()

        if norm not in ['left', 'right', 'none', 'both']:
            raise ValueError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                             ' But got "{}".'.format(norm))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self._norm = norm

        self.linear = tlx.layers.Linear(out_features=out_channels,
                                        in_features=in_channels,
                                        W_init='xavier_uniform',
                                        b_init=None)
        if add_bias is True:
            initor = tlx.initializers.Zeros()
            self.bias = self._get_weights("bias", shape=(1,self.out_channels), init=initor)

    def calculate_homophily(self, x, edge_index, het_mode='original'):
        r"""Compute the per-edge scaling factor :math:`s_{ij}` from the
        (post-linear) node features.

        Parameters
        ----------
        x: tensor
            input node feature of shape ``[num_nodes, in_channels]``.
        edge_index: tensor
            edges from src to dst, of shape ``[2, num_edges]``.
        het_mode: str, optional
            one of ``'original'``, ``'homophily'`` or ``'heterophily'``.

        Returns
        -------
        tensor
            per-edge scaling factors, of shape ``[num_edges]``.

        """
        if het_mode not in ['original', 'homophily', 'heterophily']:
            raise ValueError('Invalid het_mode value. Must be either "original", "homophily" or "heterophily".'
                             ' But got "{}".'.format(het_mode))
        num_edges = tlx.get_tensor_shape(edge_index)[1]
        if het_mode == 'original':
            return tlx.ones(shape=(num_edges, ), dtype=tlx.float32)
        src, dst = edge_index[0], edge_index[1]
        x1 = tlx.gather(x, src)
        x2 = tlx.gather(x, dst)
        # cosine similarity over the feature dimension
        dot = tlx.reduce_sum(x1 * x2, axis=-1)
        norm1 = tlx.sqrt(tlx.reduce_sum(x1 * x1, axis=-1))
        norm2 = tlx.sqrt(tlx.reduce_sum(x2 * x2, axis=-1))
        sim = dot / (norm1 * norm2 + 1e-8)
        if het_mode == 'homophily':
            return sim
        return 1.0 - sim

    def forward(self, x, edge_index, het_mode='original', edge_weight=None, num_nodes=None):
        x = self.linear(x)
        src, dst = edge_index[0], edge_index[1]
        if edge_weight is None:
            edge_weight = tlx.ones(shape=(tlx.get_tensor_shape(edge_index)[1], 1))
        edge_weight = tlx.reshape(edge_weight,(-1,))
        weights = edge_weight
        # num_nodes = tlx.reduce_max(edge_index) + 1
        if num_nodes is None:
            num_nodes = tlx.get_tensor_shape(x)[0]

        scaling = self.calculate_homophily(x, edge_index, het_mode)

        if self._norm in ['left', 'both']:
            deg = degree(src, num_nodes=num_nodes, dtype = tlx.float32)
            if self._norm == 'both':
                norm = tlx.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            weights = tlx.ops.gather(norm, src) * tlx.reshape(edge_weight, (-1,))

        if self._norm in ['right', 'both']:
            deg = degree(dst, num_nodes=num_nodes, dtype = tlx.float32)
            if self._norm == 'both':
                norm = tlx.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            weights = tlx.reshape(weights, (-1,)) * tlx.ops.gather(norm, dst)

        out = self.propagate(x, edge_index, edge_weight=weights, num_nodes=num_nodes, scaling=scaling)
        if self.add_bias:
            out += self.bias

        return out

    def message(self, x, edge_index, edge_weight=None, scaling=None):
        """
        Function that construct message from source nodes to destination nodes.

        Parameters
        ----------
        x: tensor
            input node feature.
        edge_index: tensor
            edges from src to dst.
        edge_weight: tensor, optional
            weight of each edge (the GCN normalization coefficients).
        scaling: tensor, optional
            per-edge heterophily-informed scaling factor of shape ``[num_edges]``.

        Returns
        -------
        tensor
            output message

        Returns:
            the message matrix, and the shape is [num_edges, message_dim]
        """
        msg = tlx.gather(x, edge_index[0, :])
        if edge_weight is not None:
            edge_weight = tlx.expand_dims(edge_weight, -1)
            msg = msg * edge_weight
        if scaling is not None:
            scaling = tlx.expand_dims(scaling, -1)
            msg = msg * scaling
        return msg
