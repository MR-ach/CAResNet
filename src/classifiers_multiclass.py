#they are for both 'train_g_seedx' and 'train_g_seedx_l'
import tensorflow as tf
import numpy as np
from dataset_provider import provide_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pandas as pd
import pickle

def get_one_hot(labels, nb_classes):
    res = np.eye(nb_classes)[np.array(labels).reshape(-1)]
    return np.squeeze(res.reshape(list(labels.shape)+[nb_classes]))
def sigmoid(x):
    return 1/(1 + np.exp(-x))

tf.reset_default_graph()
best_model=False
# Data input
path = './dataset/generated/randomforest/'#mac

faults_train = ['C0']
for fault_t in faults_train:
    with open(path+ fault_t+ '_train.pkl', 'rb') as f_t:
        data_t, target_t  = pickle.load(f_t)

faults_test = ['C0']
for fault_T in faults_test:
    with open(path+ fault_T+ '_test.pkl', 'rb') as f_T:
        data_T, target_T  = pickle.load(f_T)

# data_t, target_t = provide_data(path, 5600, 'train' )
# data_T, target_T = provide_data(path,1400,'test')

# Session run

# with tf.Session() as sess:
#      sess.run(tf.global_variables_initializer())
#      train_x, train_y = sess.run([data_t,target_t])#train
#      test_x, test_y = sess.run([data_T, target_T])#test

# transfer to numpy array
train_x = np.array(data_t,dtype='float32')
train_y = np.array(target_t,dtype='int64')
test_x = np.array(data_T,dtype='float32')
test_y= np.array(target_T,dtype='int64')


# from sklearn import preprocessing
# stand_means = preprocessing.StandardScaler()
# train_x = stand_means.fit_transform(train_x)
# test_x = stand_means.fit_transform(test_x)
#
repetition=50
acc_mat=[]
for re in range(repetition):
    np.random.seed(re)
    mix = [i for i in range(len(train_x))]
    np.random.shuffle(mix)
    train_x = train_x[mix]
    train_y = train_y[mix]
    # # shuffle
    # # 3,Random Forest
    cart_model = RandomForestClassifier(500,random_state=re)
    cart_model.fit(train_x, train_y)
    result = cart_model.score(test_x, test_y)
    labels_pred = cart_model.predict(test_x)
    acc=accuracy_score(test_y,labels_pred)
    acc_mat.append(acc)

r1=np.array(acc_mat)*100
print(r1)

# np.savetxt('../results/boxplot_smote.txt',r1,fmt='%.3f')

