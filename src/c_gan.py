# 导入相关库文件
import tensorflow as tf
import numpy as np
import pickle
from dataset_provider import *
import networks_c
import nextbatch

# 设置随机数进行权重初始化
seed=46777
tf.set_random_seed(seed)
np.random.seed(seed)

# 参数设置
batch_size = 2
epoch = 200
input_dim=128
z_dim=64
num_examples=10
gen_lr, dis_lr =networks_c._learning_rate()

# 加载训练数据

faults=['1','2','3']

dataset=[]
path = './dataset/C'
# labels_=[]
for fault in faults:
    with open(path + fault + '.pkl', 'rb') as f:
        data,_=pickle.load(f)
        dataset.append(data)
datasets=np.array(dataset).reshape(data.shape[0]*len(faults),data.shape[1])
# print(datasets)

label1=np.array([1]*700).reshape(700,1)
label2=np.array([2]*700).reshape(700,1)
label3=np.array([3]*700).reshape(700,1)
labels=np.vstack([label1,label2,label3])
print(datasets.shape,labels.shape)

onehot = np.eye(4) # 1维标签变成2维标签
labels=onehot[labels.astype(np.int32)].squeeze()
labels=labels[:,1:] #one_hot label
print(datasets.shape, labels.shape)
dataset_train= nextbatch.DataProvider(datasets, labels) #绑定

# tensroflow图
a=0 #lamda 设置为： 1 0.1 0.01 0.001
# 设置占位符
x = tf.placeholder(tf.float32, shape=[None, input_dim])
y = tf.placeholder(tf.float32, shape=(None, 3))
random_z = tf.random_normal([batch_size, z_dim])
# 神经网络输入与输出
g_out= networks_c.unconditional_generator_fn(random_z,y)
d_out_fake = networks_c.unconditional_discriminator_fn(g_out,y)
d_out_real = networks_c.unconditional_discriminator_fn(x,y,reuse=True)
#

# 损失设定 生成器与判别器

#WGAN损失
with tf.variable_scope('loss_D'):
    loss_D=tf.reduce_mean(d_out_fake)-tf.reduce_mean(d_out_real)
with tf.variable_scope('loss_G'):
    loss_G=-tf.reduce_mean(d_out_fake)

# # 梯度惩罚Graiednt Penalty (GP)
epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = x * epsilon + (1 - epsilon) * g_out
d_hat = networks_c.unconditional_discriminator_fn(x_hat,y, is_training=False, reuse=True)
gradients = tf.gradients(d_hat, x_hat)[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = a * tf.reduce_mean((slopes - 1.0) ** 2)
loss_D = loss_D + gradient_penalty

# # 收集变量
vars_g=[var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d=[var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

# # 优化器设置以及最小化损失设置
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')):
    opt_G_op = tf.train.AdamOptimizer(gen_lr, 0.7).minimize(loss_G,var_list=vars_g)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
    opt_D_op = tf.train.AdamOptimizer(dis_lr, 0.7).minimize(loss_D,var_list=vars_d)


# 储存路径
saver = tf.train.Saver(max_to_keep=50) #保留50个模型
train_log_dir = './model_try/model_class_'+str(a)+'/model.ckpt' #训练保存路径

# 全局设置
init = tf.global_variables_initializer() #初始化tf
config1 = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True) #查看使用的gpu

# 运行模型进行训练
with tf.Session(config=config1) as sess:
    sess.run(init)
    num_batches_per_epoch = num_examples / batch_size
    losses_G = []  # 收集generator损失值
    losses_D = []  # 收集discriminator损失值
    for i in range(epoch):
        for j in range(int(num_batches_per_epoch)):
            batch_x = dataset_train.next_batch(batch_size, shuffle=True)

            # 连续训练5次判别器才更新一次生成器 5:1
            for k in range(5):
                _, d_loss, = sess.run([opt_D_op, loss_D], feed_dict={x: batch_x[0], y: batch_x[1]})

            _, g_loss = sess.run([opt_G_op, loss_G], feed_dict={x: batch_x[0], y: batch_x[1]})
        losses_D.append(d_loss)
        losses_G.append(g_loss)

        print('epoch:', i , " d_loss: {} / g_loss: {}".format( d_loss, g_loss))
        # 每1000次保存一次模型
        if i % 1000 == 0:
            save_path = saver.save(sess, train_log_dir, global_step=i)
            print("i: {} / d_loss: {} / g_loss: {}".format(i, d_loss, g_loss))
            print("Model saved in path: %s" % save_path)

# # 储存 损失值
# # np.savetxt('../results/g_txt_' + str(a), np.array(losses_G))
# # np.savetxt('../results/d_txt_' + str(a), np.array(losses_D))

# 训练完成
print('finish training....')






