#they are for both 'train_g_seedx' and 'train_g_seedx_l'
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pandas as pd
import scipy.io as sio
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import pickle


data_path1 = './results/cnn_eca/label.txt'
data_test = np.loadtxt(data_path1)
data_test = data_test.astype(int)

data_path2 = './results/cnn_eca/pred.txt'
data_label = np.loadtxt(data_path2)
data_label = data_label.astype(int)

acc=accuracy_score(data_test,data_label)
print(acc)
cf=confusion_matrix(data_test,data_label)
print(cf)


classes = ['C0','C1','C2','C3']
# plt.figure(figsize=(3, 3),dpi=300)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes,size = 16)
plt.yticks(tick_marks, classes,size = 16)
plt.xlabel('True label',fontsize = 18)
plt.ylabel('Predict label',fontsize = 18)
# plt.show()
plt.imshow(cf) #按照像素显示出矩阵


