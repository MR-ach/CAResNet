import pickle
import tensorflow as tf
from dataset_provider import provide_data
from np_to_tfrecords import np_to_tfrecords
import numpy as np

path = './dataset/generated/'
path1 = './dataset/'
faults0 = ['C0']
for fault0 in faults0:
    with open(path1+ fault0+ '.pkl', 'rb') as f0:
        features0,labels0  = pickle.load(f0)
        # print(features0,labels0)

faults1 = ['C1']
for fault1 in faults1:
    with open(path+ fault1+ '_seed_46777.pkl', 'rb') as f1:
        features1,labels1  = pickle.load(f1)
        # print(features1,labels1)

faults2 = ['C2']
for fault2 in faults2:
    with open(path+ fault2+ '_seed_46777.pkl', 'rb') as f2:
        features2,labels2  = pickle.load(f2)
        # print(features2,labels2)

faults3 = ['C3']
for fault3 in faults3:
    with open(path+ fault3+ '_seed_46777.pkl', 'rb') as f3:
        features3,labels3  = pickle.load(f3)
        # print(features3,labels3)

features = np.vstack([features0,features1,features2,features3])
labels = np.vstack([labels0,labels1,labels2,labels3])

# 存储
# path_out = './dataset/randomforest/C0_train.pkl'
# with open(path_out, 'wb') as f:
#     pickle.dump((features,labels), f, pickle.HIGHEST_PROTOCOL)
path_out = './dataset/generated/randomforest/C0_train.pkl'
with open(path_out, 'wb') as f:
    pickle.dump((features,labels), f, pickle.HIGHEST_PROTOCOL)



# # faults = [1]
# # faults = [2]
# faults = [3]
# path = './dataset/'
# train_set = {}
# features_g = []
# labels_g = []
#
# for i in faults:
#     train_set[i] = provide_data(path, 14000, 'generated_C' + str(i) + '_seed_0')
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in faults:
#         featuresg_, labelsg_ = sess.run(train_set[i])
#         features_g.extend(featuresg_)
#         labels_g.extend(labelsg_)
#
#         features = np.array(features_g)
#         labels = np.array(labels_g,'int64')
#         print(features.shape,labels.shape)