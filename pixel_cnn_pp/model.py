"""
The core Pixel-CNN model
"""
import functools
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn


def model_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            u_list = [nn.down_shift(nn.down_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(
                        u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                query = nn.nin(nn.gated_resnet(tf.concat([ul, background], axis=3), conv=nn.nin), nr_filters)
                mixed = nn.causal_attention(key, mixin, query)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)
            # x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

            # assert len(u_list) == 0
            # assert len(ul_list) == 0

            return x_out

def pxpp_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            u_list = [nn.down_shift(nn.down_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(
                u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(
                ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(
                u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(
                ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(
                    u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(
                    ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            # /////// down pass ////////
            u = u_list.pop()
            ul = ul_list.pop()
            for rep in range(nr_resnet):
                u = nn.gated_resnet(
                    u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat(
                    [u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

            u = nn.down_shifted_deconv2d(
                u, num_filters=nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(
                ul, num_filters=nr_filters, stride=[2, 2])

            for rep in range(nr_resnet + 1):
                u = nn.gated_resnet(
                    u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat(
                    [u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

            u = nn.down_shifted_deconv2d(
                u, num_filters=nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(
                ul, num_filters=nr_filters, stride=[2, 2])

            for rep in range(nr_resnet + 1):
                u = nn.gated_resnet(
                    u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat(
                    [u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

            x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

            assert len(u_list) == 0
            assert len(ul_list) == 0

            return x_out

def h6_noup_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                query = nn.nin(nn.gated_resnet(tf.concat([ul, background], axis=3), conv=nn.nin), nr_filters)
                mixed = nn.causal_attention(key, mixin, query)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            # x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

            # assert len(u_list) == 0
            # assert len(ul_list) == 0

            return x_out

def h6_noup_hier2_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                raw_content = tf.concat([x, ul, background], axis=3)
                raw_content = tf.nn.pool(raw_content, [2, 2], "AVG", "SAME", strides=[2, 2])
                # raw_content = tf.space_to_depth(raw_content, 2)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                raw_q = tf.nn.pool(raw_q, [2, 2], "AVG", "SAME", strides=[2, 2])
                # raw_q = tf.space_to_depth(raw_q, 2)
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters)
                mixed = nn.causal_attention(key, mixin, query)

                mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, 4]), 2)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            # x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

            # assert len(u_list) == 0
            # assert len(ul_list) == 0

            return x_out

def h6_hier4_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            u_list = [nn.down_shift(nn.down_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(
                        u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hier = 4
                raw_content = tf.concat([x, ul, background], axis=3)
                raw_content = tf.nn.pool(raw_content, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                raw_q = tf.nn.pool(raw_q, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters)
                mixed = nn.causal_attention(key, mixin, query)

                mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            # x_out = nn.nin(tf.nn.elu(ul), 10 * nr_logistic_mix)

            # assert len(u_list) == 0
            # assert len(ul_list) == 0

            return x_out

def h6_1mixhier2_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            u_list = [nn.down_shift(nn.down_shifted_conv2d(
                x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(
                        u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hier = 2 if attn_rep !=0 else 1
                raw_content = tf.concat([x, ul, background], axis=3)
                if hier != 1:
                    raw_content = tf.nn.pool(raw_content, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = tf.nn.pool(raw_q, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters)
                mixed = nn.causal_attention(key, mixin, query)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out


def h8_noup_124444mix_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(8):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hiers = [1, 2, 4, 8]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h16_noup_124444mix_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(16):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hiers = [1, 2, 4, 8]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h6_noup_124mix_halfattn_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hiers = [1, 2, 4, ]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2 // 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters // 2)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h6_noup_1mix_halfattn_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hiers = [1, ]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2 // 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters // 2)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h8_noup_1mix_halfattn_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(8):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hiers = [1, ]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2 // 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters // 2)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h8_noup_1mix_halfattn_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]

                hiers = [1, ]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2 // 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters // 2)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def _base_noup_smallkey_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=256, attn_rep=12, nr_logistic_mix=10, att_downsample=1, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], axis=3)

            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(attn_rep):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]
                raw_content = tf.concat([x, ul, background], axis=3)
                q_size = 16
                raw = nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters // 2 + q_size)
                key, mixin = raw[:, :, :, :q_size], raw[:, :, :, q_size:]
                raw_q = tf.concat([ul, background], axis=3)
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), q_size)
                mixed = nn.causal_attention(key, mixin, query, downsample=att_downsample)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h6_shift_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin, nn.mem_saving_causal_shift_nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.causal_shift_nin(x_pad, nr_filters)]  # stream for up and to the left

            for attn_rep in range(6):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.mem_saving_causal_shift_nin))

                ul = ul_list[-1]

                hiers = [1, ]
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters * 2 // 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), nr_filters // 2)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

def h6_shift_small_attn_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin, nn.mem_saving_causal_shift_nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)
            ul_list = [nn.causal_shift_nin(x_pad, nr_filters)]  # stream for up and to the left

            for attn_rep in range(12):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.mem_saving_causal_shift_nin))

                ul = ul_list[-1]

                hiers = [1, ]
                attn_chns = 32
                hier = hiers[attn_rep % len(hiers)]
                raw_content = tf.concat([x, ul, background], axis=3)
                key, mixin = tf.split(nn.nin(raw_content, attn_chns * 2), 2, axis=3)
                raw_q = tf.concat([ul, background], axis=3)
                if hier != 1:
                    raw_q = raw_q[:, ::hier, ::hier, :]
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), attn_chns)
                if hier != 1:
                    key = tf.nn.pool(key, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                    mixin = tf.nn.pool(mixin, [hier, hier], "AVG", "SAME", strides=[hier, hier])
                mixed = nn.mem_saving_causal_attention(key, mixin, query, causal_unit=1 if hier == 1 else xs[2] // hier)

                if hier != 1:
                    mixed = tf.depth_to_space(tf.tile(mixed, [1, 1, 1, hier * hier]), hier)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out

h12_noup_smallkey_spec = functools.partial(_base_noup_smallkey_spec, attn_rep=12)
h12_pool2_smallkey_spec = functools.partial(_base_noup_smallkey_spec, attn_rep=12, att_downsample=2)
h8_noup_smallkey_spec = functools.partial(_base_noup_smallkey_spec, attn_rep=8)



def _base_noup_smallkey_spec_ar_chs(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=256, attn_rep=12, nr_logistic_mix=10, att_downsample=1, resnet_nonlinearity='concat_elu'):
    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], axis=3)

            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(attn_rep):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]
                raw_content = tf.concat([x, ul, background], axis=3)
                q_size = 16
                raw = nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters // 2 + q_size)
                key, mixin = raw[:, :, :, :q_size], raw[:, :, :, q_size:]
                raw_q = tf.concat([ul, background], axis=3)
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), q_size)
                mixed = nn.causal_attention(key, mixin, query, downsample=att_downsample)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            h = ul_list[-1]
            lr = nn.nin(tf.nn.elu(h), 3 * nr_logistic_mix)
            gh = nn.nin(x[:, :, :, :1], nr_filters)
            gh = nn.gated_resnet(gh, a=h, conv=nn.down_right_shifted_conv2d)
            gh = nn.gated_resnet(h, a=gh, conv=nn.down_right_shifted_conv2d)
            lg = nn.nin(tf.nn.elu(gh), 3 * nr_logistic_mix)
            bh = nn.nin(x[:, :, :, :2], nr_filters)
            bh = nn.gated_resnet(bh, a=gh, conv=nn.down_right_shifted_conv2d)
            bh = nn.gated_resnet(gh, a=bh, conv=nn.down_right_shifted_conv2d)
            lb = nn.nin(tf.nn.elu(bh), 3 * nr_logistic_mix)

            return lr, lg, lb

h8_noup_smallkey_ar_chs_spec = functools.partial(_base_noup_smallkey_spec_ar_chs, attn_rep=8)
