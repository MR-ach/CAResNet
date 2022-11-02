import pickle
import tensorflow as tf
# from dataset_provider import provide_data
# from np_to_tfrecords import np_to_tfrecords
import numpy as np

# path = './dataset/CWRU/Normal/pickle/'
# path1 = './dataset/'
# faults0 = ['Normal_COL0']
# for fault0 in faults0:
#     with open(path+ fault0+ '.pkl', 'rb') as f0:
#         features0  = pickle.load(f0)
#         print(features0)

# faults1 = ['C1']
# for fault1 in faults1:
#     with open(path+ fault1+ '_seed_46777.pkl', 'rb') as f1:
#         features1,labels1  = pickle.load(f1)
#         # print(features1,labels1)
#
# faults2 = ['C2']
# for fault2 in faults2:
#     with open(path+ fault2+ '_seed_46777.pkl', 'rb') as f2:
#         features2,labels2  = pickle.load(f2)
#         # print(features2,labels2)
#
# faults3 = ['C3']
# for fault3 in faults3:
#     with open(path+ fault3+ '_seed_46777.pkl', 'rb') as f3:
#         features3,labels3  = pickle.load(f3)
#         # print(features3,labels3)
#
# features = np.vstack([features0,features1,features2,features3])
# labels = np.vstack([labels0,labels1,labels2,labels3])
#
# # 存储
# # path_out = './dataset/randomforest/C0_train.pkl'
# # with open(path_out, 'wb') as f:
# #     pickle.dump((features,labels), f, pickle.HIGHEST_PROTOCOL)
# path_out = './dataset/generated/randomforest/C0_train.pkl'
# with open(path_out, 'wb') as f:
#     pickle.dump((features,labels), f, pickle.HIGHEST_PROTOCOL)

kernel_size = int(abs((np.math.log(16, 2) + 1) / 2))
kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
print(kernel_size)