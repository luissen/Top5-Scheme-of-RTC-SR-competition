import numpy as np

__all__ = ['handlers']


def addmm(node):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p


def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    return n * m


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    return b * n * m * p


def matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return n
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return np.prod(b) * n * m
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return np.prod(b) * n * m
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return np.prod(b) * n * m * p


def mul(node):
    os = node.outputs[0].shape
    return np.prod(os)


def convolution(node):
    os = node.outputs[0].shape
    oc, ic, *ks = node.inputs[1].shape
    flops = np.prod(os) * ic * np.prod(ks)
    return flops


def batch_norm(node):
    os = node.outputs[0].shape
    flops = np.prod(os) * 4
    return flops


def instance_norm_or_layer_norm(node):
    os = node.outputs[0].shape
    return np.prod(os)


def avg_pool_or_mean(node):
    os = node.outputs[0].shape
    return np.prod(os)

def relu(node):
    os = node.outputs[0].shape
    flops = np.prod(os)
    return flops

def adp_max_pool(node):
    oc, ic, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    flops = np.prod(ks) * np.prod(os)
    return flops

def convtranspose(node):
    oc, ic, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    flops = np.prod(os) * np.prod(ks) * oc * ic
    return flops

def softmax(node):

    b, c, h,w = node.outputs[0].shape
    features = c * h * w
    exp = features
    add = features - 1
    div = features
    flops = b * (exp + add + div)

    return flops

def pixel_shuffle(node):
    os = node.outputs[0].shape
    flops = np.prod(os)
    return flops

def linear(node):
    os = node.inputs[1].shape
    mul = np.prod(os)
    add = mul - 1
    out_os = node.outputs[0].shape
    flops = (mul + add) * out_os
    return flops

def sigmoid(node):

    os = node.inputs[0].shape
    nelements = np.prod(os)
    exp = nelements
    add = nelements
    div = nelements

    flops = exp + add + div
    return flops

handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    ('aten::matmul', matmul),
    (('aten::mul', 'aten::mul_'), mul),
    ('aten::_convolution', convolution),
    ('aten::batch_norm', batch_norm),
    (('aten::instance_norm', 'aten::layer_norm'), instance_norm_or_layer_norm),
    (('aten::adaptive_avg_pool1d', 'aten::adaptive_avg_pool2d', 'aten::adaptive_avg_pool3d',
      'aten::avg_pool1d', 'aten::avg_pool2d', 'aten::avg_pool3d', 'aten::mean'), avg_pool_or_mean),
    (('aten::relu', 'aten::relu_'), relu),
    (('aten::adaptive_max_pool1d', 'aten::adaptive_max_pool2d', 'aten::adaptive_max_pool3d'), adp_max_pool),
    (('aten::transpose'), convtranspose),
    (('aten::softmax'),softmax),
    ('aten::pixel_shuffle', pixel_shuffle),
    ('aten:linear', linear),
    ('aten::sigmoid', sigmoid),

    (('aten::add', 'aten::add_',
      'aten::alpha_dropout', 'aten::cat', 'aten::chunk', 'aten::clone', 'aten::constant_pad_nd', 'aten::contiguous',
      'aten::div', 'aten::div_', 'aten::dropout', 'aten::dropout_', 'aten::embedding', 'aten::eq',
      'aten::feature_dropout', 'aten::flatten', 'aten::gt', 'aten::hardtanh_', 'aten::int', 'aten::lt',
      'aten::log_softmax', 'aten::max_pool1d', 'aten::max_pool1d_with_indices', 'aten::max_pool2d',
      'aten::max_pool2d_with_indices', 'aten::max_pool3d', 'aten::max_pool3d_with_indices', 'aten::max_unpool1d',
      'aten::max_unpool2d', 'aten::max_unpool3d', 'aten::ne', 'aten::reflection_pad1d', 'aten::reflection_pad2d',
      'aten::reflection_pad3d', 'aten::replication_pad1d', 'aten::replication_pad2d',
      'aten::replication_pad3d', 'aten::select',  'aten::size', 'aten::slice',
      'aten::softshrink', 'aten::sub', 'aten::sum', 'aten::t', 'aten::tanh', 'aten::threshold',
      'aten::view', 'prim::constant', 'prim::listconstruct', 'prim::listunpack', 'prim::numtotensor'), None)
)
