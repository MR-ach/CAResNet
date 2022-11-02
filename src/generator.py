import dataset_provider
import networks_c
import nextbatch
from np_to_tfrecords import *
import tensorflow as tf
import pickle
import numpy as np

# lamda = ['1', '0.1', '0.01', '0.001', '0']
lamda = [ '1']
# faults = '1'
# faults = '2'
faults = '3'
# faults = '1','2','3'
dataset = []
labels_ = []
for a in lamda:

    # 数据加载
    # tf.compat.v1.reset_default_graph()
    tf.reset_default_graph()
    path = './dataset/C'
    for fault in faults:
        with open(path + fault + '.pkl', 'rb') as f:
            data, _ = pickle.load(f)
    labels = np.repeat([np.array([1, 0, 0])], 700, axis=0)
    print(labels)

    # 参数设置
    batch_size = 2
    input_dim = 128
    z_dim = 64
    gen_lr, dis_lr = networks_c._learning_rate()
    epoch = 10000

    # ---------------------------重建之前的tensroflow graph-----------------------------------------
    # graph
    x = tf.placeholder(tf.float32, shape=[None, input_dim])
    y = tf.placeholder(tf.float32, shape=(None, 3))
    random_z = tf.placeholder(tf.float32, shape=[None, z_dim])

    # draw a graph of GAN model
    g_out = networks_c.unconditional_generator_fn(random_z, y)
    d_out_fake = networks_c.unconditional_discriminator_fn(g_out, y)
    d_out_real = networks_c.unconditional_discriminator_fn(x, y, reuse=True)

    # loss graph(improved Wgan)
    with tf.variable_scope('loss_D'):
        loss_D = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
    with tf.variable_scope('loss_G'):
        loss_G = -tf.reduce_mean(d_out_fake)

    # gradient penalty
    epsilon = tf.random_uniform([], 0.0, 1.0)
    x_hat = x * epsilon + (1 - epsilon) * g_out
    d_hat = networks_c.unconditional_discriminator_fn(x_hat, y, is_training=False, reuse=True)
    gradients = tf.gradients(d_hat, x_hat)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = 1 * tf.reduce_mean((slopes - 1.0) ** 2)
    loss_D += gradient_penalty

    # variables for gen and dis training
    vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

    # optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
        opt_G_op = tf.train.AdamOptimizer(gen_lr, 0.7).minimize(loss_G, var_list=vars_g)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
        opt_D_op = tf.train.AdamOptimizer(dis_lr, 0.7).minimize(loss_D, var_list=vars_d)


    # --------------------------------------------------------------------------------------------------

    # 生成器的输出
    eval_examples = networks_c.unconditional_generator_fn(
        tf.random_normal([batch_size, z_dim]), y,
        is_training=False, reuse=True)

    # 保存
    saver = tf.train.Saver()

    # 全局设置
    global_step = tf.train.get_or_create_global_step()
    init = tf.global_variables_initializer()
    config1 = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

    # 控制输出
    y_slect = labels[0:2]
    num_examples_to_eval = 700
    name_generated = 'generated_C' + faults + '_seed_' + a

    with tf.Session() as sess:
        # 加载保存的模型
        new_saver = tf.train.import_meta_graph(
            './model_try/model_class_' + a + '/model.ckpt-19000.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(
            './model_try/model_class_' + str(a)))

        # 进行数据生成
        count_examples = 0
        generated_examples = []
        while count_examples < num_examples_to_eval:
            sample = sess.run(eval_examples, feed_dict={y: y_slect})
            sample = np.squeeze(sample)
            generated_examples.extend(sample)
            count_examples += batch_size
        generated_examples = np.array(generated_examples)
        generated_labels = np.repeat([[int(faults)]], generated_examples.shape[0], axis=0)
        generated_labels = np.array(generated_labels, dtype=np.int64)

    print(generated_examples.shape, generated_labels.shape)  # 打印shape

    # 保存数据
    np_to_tfrecords(generated_examples, generated_labels,'./dataset/imbalance/' + name_generated)

