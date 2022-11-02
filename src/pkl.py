import pickle
import tensorflow as tf
from dataset_provider import provide_data
from np_to_tfrecords import np_to_tfrecords
import numpy as np

# path='F:/paper code/paper 1 code/rf_windturbine/dataset/wpt/balance/'
#
# # 读取正常数据
#
# faults = ['C0']
# for fault in faults:
#     with open(path+ fault+ '_b.tfrecords', 'rb') as f:
#         raw_dataset = tf.data.TFRecordDataset(f)

# faults = [0,1,2,3]
faults = [3]
path = './dataset/generated/'
path1 = "F:/paper code/paper 1 code/rf_windturbine/dataset/wpt/generated/"
train_set = {}
features_g = []
labels_g = []

for i in faults:
    train_set[i] = provide_data(path1, 14000, 'generated_C' + str(i) + '_seed_46777')
    # if i<1:
    #     train_set[i] = provide_data(path, 14000, 'wpt/balance/C' + str(i) + '_b')
    # else:
    #     train_set[i] = provide_data(path, 140, 'wpt/imbalance/C' + str(i))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in faults:
        featuresg_, labelsg_ = sess.run(train_set[i])
        features_g.extend(featuresg_)
        labels_g.extend(labelsg_)

        features = np.array(features_g)
        labels = np.array(labels_g,'int64')
        print(features.shape,labels.shape)

        with open(path + 'C' + str(i) + '_seed_46777.pkl','wb') as f:
            pickle.dump([features,labels ],f,pickle.HIGHEST_PROTOCOL)
    # with open('F:/paper code/exercise/1dcnn/dataset/C1.pkl','wb') as f:
    #     pickle.dump([features,labels],f,pickle.HIGHEST_PROTOCOL)
    # with open('F:/paper code/exercise/1dcnn/dataset/C2.pkl','wb') as f:
    #     pickle.dump([features,labels],f,pickle.HIGHEST_PROTOCOL)
    # with open('F:/paper code/exercise/1dcnn/dataset/C3.pkl','wb') as f:
    #     pickle.dump([features,labels],f,pickle.HIGHEST_PROTOCOL)

# C0 = train_set[0]
# C0 =C0[0]
# # C0 =C0.shape
# C1 = train_set[0]
# C1 =C1[0]
# # C1 =C1.shape
# C2 = train_set[0]
# C2 =C2[0]
# # C2 =C2.shape
# C3 = train_set[0]
# C3 =C3[0]
# # C3 =C3.shape
# print(C0)
# print(C1)
# print(C2)
# print(C3)





#     for i in faults:
#         featuresg_, labelsg_ = sess.run(train_set[i])
#         features_g.extend(featuresg_)
#         labels_g.extend(labelsg_)
#
# features = np.array(features_g)
# labels = np.array(labels_g,'int64')
# print(features.shape,labels.shape)

# from collections import Counter
#
# y=labels.reshape(730,)
# counter = Counter(y)
# print(counter)
#
# from imblearn.over_sampling import SMOTE
#
# oversample=SMOTE(sampling_strategy='auto',random_state=None,k_neighbors=7)
# train_x,train_y=oversample.fit_resample(features,y)
# counter = Counter(train_y)
# print(counter)
# # train_y = train_y.reshape(2800, 1)
# # np_to_tfrecords(train_x, train_y, '../dataset/wpt/fitting/train_smote', verbose=True)
