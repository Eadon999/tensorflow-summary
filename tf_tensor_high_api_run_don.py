import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from layer.interaction import BiInteractionPooling
from tensorflow.python.keras import backend as K



class BiInteractionPooling(Layer):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.
      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """

    def __init__(self, **kwargs):

        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        super(BiInteractionPooling, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term


class MyLayer(Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, unit),
                                      initializer=tf.keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(unit,),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, input_flatten, **kwargs):
        fm_input = tf.expand_dims(input_flatten, axis=1)  # (3, 1, 5)

        # 打印张量的值方法一：
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 必须
            print(sess.run(fm_input))
        # 打印张量的值方法二：先在外部调用前tf.enable_eager_execution()，之后直接运行
        # print(fm_input)
        print(fm_input.shape)

        fm_logit = BiInteractionPooling()(fm_input)  # 二阶段交叉
        print(fm_logit.shape)

        y1 = tf.layers.dense(inputs=input_flatten, name='layer1', units=100, activation=tf.nn.relu,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        dnn_logit = tf.layers.dense(inputs=y1, name='layer2', units=50, activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        # (3, 50)
        print(dnn_logit.shape)

        #
        fm_logit_reshape = tf.squeeze(fm_logit, [1])

        print(fm_logit_reshape.shape)
        concat_layer = tf.concat([fm_logit_reshape, dnn_logit], axis=1)
        print(concat_layer.shape)

        logits = tf.layers.dense(inputs=concat_layer, name='output', units=1, activation=None,
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        return logits


if __name__ == '__main__':
    # 打印张量的值方法一：
    # tf.enable_eager_execution()
    ##在1.15也可以直接使用
    print(tf.version.VERSION)
    x = tf.ones((3, 5))
    my_layer = MyLayer(5, 2)
    out = my_layer(x)
    print(out)

    # 打印张量的值方法二：
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 必须
        print(sess.run(out))
'''
以上程序输出结果
1.15.2
WARNING:tensorflow:From /Volumes/Files/Code/Git_Hub/Company/tfboy/example/live_rank_example/test_tf.py:10: The name tf.keras.initializers.RandomNormal is deprecated. Please use tf.compat.v1.keras.initializers.RandomNormal instead.

WARNING:tensorflow:From /Users/don/.pyenv/versions/3.7.4/lib/python3.7/site-packages/tensorflow_core/python/keras/initializers.py:143: calling RandomNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /Users/don/.pyenv/versions/3.7.4/lib/python3.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /Users/don/.pyenv/versions/3.7.4/lib/python3.7/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From /Users/don/.pyenv/versions/3.7.4/lib/python3.7/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

2021-04-26 00:44:06.716988: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2021-04-26 00:44:06.732666: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fd23a4282c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-04-26 00:44:06.732696: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[[[1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]]]
(3, 1, 5)
WARNING:tensorflow:From /Volumes/Files/Code/Git_Hub/Company/tfboy/layer/utils.py:159: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /Volumes/Files/Code/Git_Hub/Company/tfboy/example/live_rank_example/test_tf.py:31: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.Dense instead.
WARNING:tensorflow:From /Users/don/.pyenv/versions/3.7.4/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
(3, 1, 5)
(3, 50)
(3, 5)
(3, 55)
Tensor("my_layer/output/BiasAdd:0", shape=(3, 1), dtype=float32)
[[0.19000974]
 [0.19000974]
 [0.19000974]]

Process finished with exit code 0

'''
