import os
import random
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

#check version
print('numpy',np.__version__)
print('tensorflow',tf.__version__)
# !python --version
def set_seed(seed):

  # os.environ['PYTHONHASHSEED']=str(0)
  random.seed(seed)
  tf.random.set_seed(seed)
  # tf.keras.utils.set_random_seed(seed)
  tf.compat.v1.set_random_seed(seed)
  np.random.seed(seed)
  # os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'

set_seed(920)

# self define
#load data
def LoadData_pickle_1(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data,label=pickle.load(f)

  return data,label

def LoadData_pickle_T(path,name,type='rb'):
  with open(path+name+'_test.pkl', type) as f:
          data,label=pickle.load(f)

  return data,label

# TF data db
def TFData_preprocessing(x,y,batch_size,conditional=True):
  if conditional:
      x=tf.data.Dataset.from_tensor_slices((x,y))
      x=x.shuffle(10000).batch(batch_size)
  else:
      x=tf.data.Dataset.from_tensor_slices(x)
      x=x.shuffle(10000).batch(batch_size)

  return x

def one_hot_MPT(y,depth):
# #one-hot
  y=tf.one_hot(y,depth=depth, dtype=tf.int32)
  # y=y[:,1:]# 前面一列删除

  return y
# metric
def accuracy(train_x, train_y, test_x, test_y):
  cart_model = RandomForestClassifier(100)
  cart_model.fit(train_x, train_y)
  labels_pred = cart_model.predict(test_x)
  current_acc = accuracy_score(test_y, labels_pred) * 100

  return current_acc


from sklearn import svm
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_train(dir, fault):
    path_train = dir + fault + '.pkl'
    with open(path_train, 'rb') as f:
        X_train, _ = pickle.load(f)
    return X_train

def load_test(dir, fault):
    path_train = dir + fault + '_test.pkl'
    with open(path_train, 'rb') as f:
        _, X_test = pickle.load(f)
    return X_test

def two_d_data(x,axis):
    x_ = np.expand_dims(x, axis=axis)
    return x_

def plot(x):
    plt.plot(x[100:200, :].T, 'b')
    plt.show()





from tensorflow import keras
import tensorflow as tf
class InstanceNormalization(keras.layers.Layer):
    def __init__(self, axis=(1, 2), epsilon=1e-6):
        super().__init__()
        # NHWC
        self.epsilon = epsilon
        self.axis = axis
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        # NHWC
        shape = [1,  1, input_shape[-1]]
        self.gamma = self.add_weight(
            name='gamma',
            shape=shape,
            initializer='ones')

        self.beta = self.add_weight(
            name='beta',
            shape=shape,
            initializer='zeros')

    def call(self, x, *args, **kwargs):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        diff = x - mean
        variance = tf.reduce_mean(tf.math.square(diff), axis=self.axis, keepdims=True)
        x_norm = diff * tf.math.rsqrt(variance + self.epsilon)
        return x_norm * self.gamma + self.beta
# dai ma
# Load source domain datasets
fault = 'C0'

x, y = LoadData_pickle_1(path='./dataset/',
                       name=fault)
print(x.shape,y.shape)
# y = np.array(y).reshape(x.shape[0], )
# print(x.shape[0])
y = np.array(y).reshape(x.shape[0], )
# print(y)
y_A = one_hot_MPT(y, depth=4)
# print(y_A)

mix = [i for i in range(len(x))]
np.random.shuffle(mix)
x_A = x[mix]
# print('--Source Domain--')
print('Feature shape:', x_A.shape, 'Label shape', y_A.shape, )
print('Label number are:', y_A[0].numpy(), 'for class 0 (normal)')

print()
# Load target domain datasets
faults1 = ['C1', 'C2', 'C3']
data_x = []
data_y = []
for fault1 in faults1:
    x, y = LoadData_pickle_1(path='./dataset/',
                           name=fault1)
    data_x.extend(x)
    data_y.extend(y)

x_B = np.array(data_x)
data_y=np.array(data_y).reshape(x_B.shape[0],)
print(x_B.shape,data_y.shape)
data_y = np.array([1] * 700 + [2] * 700 + [3] * 700)
# print(data_y.shape)
y_B = one_hot_MPT(data_y, depth=4)


# print('--Target Domain--')
print('Feature shape:', x_B.shape, 'Label shape', y_B.shape, )

print('Label number are:', y_B[0].numpy(), 'for class 0 (normal)', y_B[700].numpy(), 'for class 0 (normal)',
      y_B[1400].numpy(), 'for class 0 (normal)')

# load test data
faults2 = ['C0', 'C1', 'C2', 'C3']
data_x_T = []
data_y_T = []
for fault2 in faults2:
    x, _ = LoadData_pickle_T(path='./dataset/',
                             name=fault2)
    data_x_T.extend(x)

data_x_T = np.array(data_x_T)
data_y_T = np.array([0] * 300 + [1] * 300 + [2] * 300 + [3] * 300)

data_x = np.vstack((x_A, x_B))   # 数
data_y = np.vstack((y_A, y_B))   # 维度改成数y_A, y_B
# print(data_y)
print(data_x.shape, data_y.shape)
print(data_x_T.shape, data_y_T.shape)

train_db = TFData_preprocessing(data_x, data_y, batch_size=32)
test_db = TFData_preprocessing(data_x_T, data_y_T, batch_size=32)

# model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense, Concatenate, Input, \
    Reshape, ReLU, Conv1D, Conv1DTranspose, BatchNormalization, GlobalAveragePooling1D, multiply, \
    GlobalAveragePooling2D, \
    GlobalMaxPooling1D, Add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)

af = []


# 归一化
def do_norm(norm):
    if norm == "batch":
        _norm = BatchNormalization()
    elif norm == "instance":
        _norm = InstanceNormalization()
    else:
        _norm = []
    return _norm


# 卷积
def gen_block_down(filters, k_size, strides, padding, input, norm="instance"):
    g = Conv1D(filters, k_size, strides=strides, padding=padding)(input)
    g = do_norm(norm)(g)
    g = ReLU(0.3)(g)
    return g


def gen_block_down_1(filters, k_size, strides, padding, input, norm="batch"):
    g = Conv1D(filters, k_size, strides=strides, padding=padding)(input)
    g = do_norm(norm)(g)
    g = ReLU(0.3)(g)
    return g


# 注意力机制SE
def se_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    se_feature = GlobalAveragePooling1D()(input_feature)
    se_feature = Reshape((1, channel))(se_feature)

    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=False,
                       bias_initializer='zeros'
                       )(se_feature)

    se_feature = Dense(channel,
                       kernel_initializer='he_normal',
                       use_bias=False,
                       bias_initializer='zeros'
                       )(se_feature)
    se_feature = Activation('sigmoid')(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature


# 注意力机制ECA
def eca_block_1(input_feature, b=1, gamma=2):
    channel = input_feature.shape[-1]
    kernel_size = int(abs((np.math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = GlobalAveragePooling1D()(input_feature)

    x = Reshape((-1, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False, )(x)
    x = Activation('sigmoid')(x)
    # x = Activation('relu')(x)
    x = Reshape((1, -1))(x)
    # x = x.reshape(None, 1536, 16)

    output = multiply([input_feature, x])
    return output


def eca_block_2(input_feature_2, b=1, gamma=2, name=""):
    channel = input_feature_2.shape[-1]
    kernel_size = int(abs((np.math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    avg_pool = GlobalAveragePooling2D()(input_feature_2)

    x = Reshape((-1, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_1_" + str(name), use_bias=False, )(x)
    # x = Activation('sigmoid')(x)
    x = Activation('relu')(x)
    x = Reshape((1, -1))(x)

    output = multiply([input_feature_2, x])
    return output


# resnet
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # first layer convolutional laycer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # second convolutional layer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


def modified_Resnet_eca(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)

    # first layer convolutional laycer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # second convolutional layer
    g = Conv1D(n_filters, 5, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    # se
    # print('asdasd1')
    se = se_block(g)
    # print('asdasd2')
    se = InstanceNormalization(axis=-1)(se)

    out = Concatenate()([se, input_layer])

    return out

    # concatenate merge channel-wise with input layer
    # g = Concatenate()([g, input_layer])


class CNN(keras.Model):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.cnn_model = self._get_1dcnn()

        self.opt = keras.optimizers.Adam(1e-4)
        # self.opt=keras.optimizers.RMSprop(1e-4)
        # self.opt=keras.optimizers.Adagrad(1e-4)
        self.loss_bool = keras.losses.CategoricalCrossentropy(from_logits=True)

    def _get_1dcnn(self):

        # weight initialization
        # init = RandomNormal(stddev=0.02)
        in_image = Input(shape=self.img_shape)

        x = Reshape((3072, 1))(in_image)  # None,1,1024,1

        enc = gen_block_down_1(32, 5, 2, 'same', x)
        enc = gen_block_down_1(16, 5, 2, 'same', enc)

        enc = gen_block_down(8, 5, 2, 'same', enc)
        enc = gen_block_down(16, 5, 2, 'same', enc)

        enc = gen_block_down(32, 5, 2, 'same', enc)
        enc = gen_block_down(64, 5, 2, 'same', enc)

        enc = Flatten()(enc)
        logits = keras.layers.Dense(4, activation='softmax')(enc)

        model = keras.models.Model(inputs=in_image, outputs=logits)
        model.summary()

        return model

    def train_loss(self, x, y):
        with tf.GradientTape() as tape:
            pred_y = self.cnn_model(x)
            loss = self.loss_bool(y, pred_y)

        var = self.cnn_model.trainable_variables
        dis_grads = tape.gradient(loss, var)
        self.opt.apply_gradients(zip(dis_grads, var))
        return loss

    @tf.function
    def train_on_step(self, image_batch):
        images, labels = image_batch
        loss = self.train_loss(images, labels)
        return loss

    # @tf.function
    def train(self, train_db, test_db, epochs):
        for epoch in range(epochs):
            for images_batch in train_db:
                images, image_labels = images_batch
                loss = self.train_on_step(images_batch)

            # evaluate on test
            correct, total = 0, 0
            pred = []
            label = []
            for x_T, y_T in test_db:
                pred_y_T = self.cnn_model(x_T)
                pred_y_T = tf.argmax(pred_y_T, axis=-1)
                # print(pred_y_T)
                y_T = tf.cast(y_T, tf.int64)
                correct += float(tf.reduce_sum(tf.cast(tf.equal(pred_y_T, y_T), tf.float32)))
                # print(pred_y_T)
                total += x_T.shape[0]
                acc = correct / total
                pred.extend(pred_y_T)
                label.extend(y_T)
            af.append(acc)
            los.append(loss.numpy())

            print('epoch:', epoch, 'loss:', loss.numpy(), 'test acc:', correct / total)
        mean = np.mean(af)
        print('avg accuracy=',mean)

        pred = np.array(pred)
        pred = pred.reshape(-1, 1)
        label = np.array(label)
        label = label.reshape(-1, 1)
        # np.savetxt ('./results/cnn/cnn_650.txt',af)
        # np.savetxt('./results/cnn/cnn_loss_650.txt', los)
        np.savetxt('./results/cnn/pred.txt', pred)
        np.savetxt('./results/cnn/label.txt', label)
        print(pred)
        print(label)


# Run
af = []
los = []

input_shape = (3072,)
cnn = CNN(input_shape)
cnn.train(train_db, test_db, epochs=24)




