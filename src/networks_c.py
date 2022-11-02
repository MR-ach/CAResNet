
  #for c_gan
import tensorflow as tf

framework = tf.contrib.framework
layers = tf.contrib.layers
ds = tf.contrib.distributions

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)


def _learning_rate():
    generator_lr = tf.train.exponential_decay(
        learning_rate=1e-4,
        global_step=tf.train.get_or_create_global_step(),
        decay_steps=50000,
        decay_rate=0.9,
        staircase=True)
    discriminator_lr = 1e-5
    return generator_lr, discriminator_lr


def _optimizer(gen_lr, dis_lr, use_sync_replicas):
    """Get an optimizer, that's optionally synchronous."""
    generator_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9, momentum=0.1)
    discriminator_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95, momentum=0.1)
    return generator_opt, discriminator_opt


def unconditional_generator_fn(noise, y,weight_decay=2.5e-5, is_training=True,reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        with framework.arg_scope([layers.fully_connected, layers.conv2d_transpose],activation_fn=_leaky_relu):
            cat1=tf.concat([noise,y],1)
            net = layers.fully_connected(cat1, 1024)
            net = layers.linear(net, 128, normalizer_fn=None)
            return net


def unconditional_discriminator_fn(img,y,weight_decay=2.5e-5, is_training=True,reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.contrib.framework.arg_scope([layers.conv2d, layers.fully_connected],activation_fn=_leaky_relu):

            cat1 = tf.concat([img, y], 1)
            net = layers.fully_connected(cat1, 1024)
            net = layers.fully_connected(net, 2048)
            return layers.linear(net, 1)



