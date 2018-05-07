# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layer.reorg_layer import reorg_layer

_activation_dict = {
    'leaky': tf.nn.leaky_relu,
    'relu': tf.nn.relu
}


# cfg_layerName(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
#    pass

def cfg_net(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    width = int(param["width"])
    height = int(param["height"])
    channels = int(param["channels"])
    net = tf.placeholder(tf.float32, [None, width, height, channels], name=scope)
    return net


def cfg_convolutional(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    batch_normalize = 'batch_normalize' in param
    size = int(param['size'])
    filters = int(param['filters'])
    stride = int(param['stride'])
    pad = 'same' if param['pad'] == '1' else 'valid'
    activation = None
    weight_size = C * filters * size * size

    if "activation" in param:
        activation = _activation_dict.get(param['activation'], None)

    biases, scales, rolling_mean, rolling_variance, weights = \
        weights_walker.get_weight(param['name'],
                                  filters=filters,
                                  weight_size=weight_size,
                                  batch_normalize=batch_normalize)
    weights = weights.reshape(C, filters, size, size).transpose([2, 3, 1, 0])

    conv_args = {
        "filters": filters,
        "kernel_size": size,
        "strides": stride,
        "activation": activation,
        "kernel_initializer": tf.initializers.constant(weights),
        "bias_initializer": tf.initializers.constant(biases),
        "padding": pad,
    }
    net = tf.layers.conv2d(net, name=scope, **conv_args)

    if batch_normalize:
        batch_norm_args = {
            "gamma_initializer": tf.initializers.constant(scales),
            "moving_mean_initializer": tf.initializers.constant(rolling_mean),
            "moving_variance_initializer": tf.initializers.constant(rolling_variance),
            "fused": False,
            "trainable": False,
        }
        net = tf.layers.batch_normalization(net, name=scope, **batch_norm_args)

    return net


def cfg_maxpool(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    pool_args = {
        "pool_size": int(param['size']),
        "strides": int(param['stride']),
    }
    net = tf.layers.max_pooling2d(net, name=scope, **pool_args)
    return net


def cfg_route(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    if not isinstance(param["layers"], list):
        param["layers"] = [param["layers"]]
    net_index = [int(x) for x in param["layers"]]
    nets = [stack[x] for x in net_index]

    net = tf.concat(nets, axis=-1, name=scope)
    return net


def cfg_reorg(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    reorg_args = {
        "stride": int(param['stride'])
    }

    net = reorg_layer(net, name=scope, **reorg_args)
    return net

def cfg_shortcut(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    index = int(param['from'])
    activation = param['activation']
    assert activation == 'linear'

    from_layer = stack[index]
    net = tf.add(net, from_layer, name=scope)
    return net

def cfg_yolo(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    output_index.append(len(stack) - 1)
    return net

def cfg_upsample(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    stride = int(param['stride'])
    assert stride == 2

    net = tf.image.resize_nearest_neighbor(net, (H * stride, W * stride), name=scope)
    return net

def cfg_region(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    pass

def cfg_ignore(B, H, W, C, net, param, weights_walker, stack, output_index, scope=None):
    print("=> Ignore: ", param)

    return net


_cfg_layer_dict = {
    "net": cfg_net,
    "convolutional": cfg_convolutional,
    "maxpool": cfg_maxpool,
    "route": cfg_route,
    "reorg": cfg_reorg,
    "shortcut": cfg_shortcut,
    "yolo": cfg_yolo,
    "upsample": cfg_upsample,
    "region": cfg_region,
}


def get_cfg_layer(net, layer_name, param, weights_walker, stack, output_index, scope=None):
    B, H, W, C = [None, None, None, None] if net is None else net.shape.as_list()
    layer = _cfg_layer_dict.get(layer_name, cfg_ignore)(B, H, W, C, net, param, weights_walker, stack, output_index, scope)
    return layer
