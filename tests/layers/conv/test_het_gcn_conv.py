# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2026/11/25
# @Author  : GammaGL contributor
# @FileName: test_het_gcn_conv.py

import numpy as np
import pytest
import tensorlayerx as tlx
from gammagl.layers.conv import GCNConv, HetGCNConv


def _copy_weights(src_conv, dst_conv):
    """Copy all trainable weights from src_conv into dst_conv."""
    weights = [tlx.convert_to_numpy(w) for w in src_conv.all_weights]
    tlx.files.assign_weights(weights, dst_conv)


def test_het_gcn_conv_reduction_to_gcn():
    # het_mode='original' (s_ij = 1) must reduce exactly to a plain GCNConv
    # with the same weights on the same graph.
    np.random.seed(0)
    x_np = np.random.randn(4, 16).astype(np.float32)
    ei_np = np.array([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]], dtype=np.int64)
    x = tlx.convert_to_tensor(x_np)
    edge_index = tlx.convert_to_tensor(ei_np)

    gcn = GCNConv(in_channels=16, out_channels=16)
    het = HetGCNConv(in_channels=16, out_channels=16)
    _copy_weights(gcn, het)

    out_gcn = tlx.convert_to_numpy(gcn(x, edge_index))
    out_het = tlx.convert_to_numpy(het(x, edge_index, het_mode='original'))

    assert out_gcn.shape == (4, 16)
    assert out_het.shape == (4, 16)
    assert np.allclose(out_het, out_gcn, atol=1e-5)


def test_het_gcn_conv_modes_run_and_differ():
    np.random.seed(1)
    x_np = np.random.randn(5, 8).astype(np.float32)
    ei_np = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    x = tlx.convert_to_tensor(x_np)
    edge_index = tlx.convert_to_tensor(ei_np)

    conv = HetGCNConv(in_channels=8, out_channels=8)
    out_original = tlx.convert_to_numpy(conv(x, edge_index, het_mode='original'))
    out_homophily = tlx.convert_to_numpy(conv(x, edge_index, het_mode='homophily'))
    out_heterophily = tlx.convert_to_numpy(conv(x, edge_index, het_mode='heterophily'))

    for out in (out_original, out_homophily, out_heterophily):
        assert out.shape == (5, 8)
        assert np.all(np.isfinite(out))

    # random features give non-uniform cosine similarities, so both modulated
    # modes must deviate from the plain-GCN output
    assert not np.allclose(out_homophily, out_original, atol=1e-5)
    assert not np.allclose(out_heterophily, out_original, atol=1e-5)


def test_het_gcn_conv_invalid_mode():
    x = tlx.random_normal(shape=(4, 8))
    edge_index = tlx.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype=tlx.int64)
    conv = HetGCNConv(in_channels=8, out_channels=8)
    with pytest.raises(ValueError):
        conv(x, edge_index, het_mode='not-a-mode')


def test_het_gcn_conv_permutation_equivariance():
    # permuting the node order (relabel edge_index + permute x) must permute
    # the output rows identically: with x2 = x[perm], out2[perm] == out
    np.random.seed(2)
    x_np = np.random.randn(6, 8).astype(np.float32)
    ei_np = np.array([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 3]], dtype=np.int64)
    perm = np.array([5, 3, 1, 0, 2, 4])
    inv = np.argsort(perm)

    conv = HetGCNConv(in_channels=8, out_channels=8)
    out = tlx.convert_to_numpy(conv(tlx.convert_to_tensor(x_np),
                                    tlx.convert_to_tensor(ei_np),
                                    het_mode='heterophily'))
    out2 = tlx.convert_to_numpy(conv(tlx.convert_to_tensor(x_np[perm]),
                                     tlx.convert_to_tensor(inv[ei_np]),
                                     het_mode='heterophily'))

    assert np.allclose(out2[perm], out, atol=1e-5)


def test_het_gcn_conv_gradient_flow():
    np.random.seed(3)
    x = tlx.convert_to_tensor(np.random.randn(4, 8).astype(np.float32))
    edge_index = tlx.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype=tlx.int64)

    conv = HetGCNConv(in_channels=8, out_channels=8)
    out = conv(x, edge_index, het_mode='homophily')
    loss = tlx.reduce_mean(out)

    optimizer = tlx.optimizers.Adam(lr=1e-3)
    grads = optimizer.gradient(loss, conv.trainable_weights)
    for grad in grads:
        assert grad is not None
        arr = tlx.convert_to_numpy(grad)
        assert np.all(np.isfinite(arr))
