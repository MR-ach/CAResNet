# 随机森林
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#以下为非batchnorm 操作后的生成数据
# c_g_acc为 传统的wgan
c_g= np.array(    [98.44,98.10,98.41,98.35,98.01,98.28,98.00,98.43,98.51,98.20,98.07,98.15,98.18,98.43,98.58,
               98.46,98.63,97.98,98.12,98.30,98.63,98.38,98.38,98.28,98.42,98.44,98.31,97.80,97.93,98.47])
#c_g 为传统的wgan 最后一次迭代
c_g_acc=     np.array(      [98.28,98.63,98.37,98.03,98.33,98.29,98.28,98.31,98.17,98.30,98.24,98.01,97.08,98.35,98.48,
                97.57,98.62,98.58,98.48,98.46,98.27,98.50,98.30,98.28,98.60,98.11,98.37,98.18,97.28,98.40])

# c_g_acc_mse为wgan+mse损失
c_g_acc_mse= np.array( [98.45,98.58,98.68,98.23,97.25,98.45,97.90,98.08,98.26,98.03,98.40,98.58,97.50,98.49,98.50,
               98.20,98.74,98.61,98.36,98.60,98.49,98.48,98.42,98.49,98.53,98.45,98.52,98.45,97.43,98.42])
# c_g_mse 为wgan+mse损失的最后一次迭代
c_g_mse=     np.array(  [98.30,98.65,98.45,98.48,98.26,98.47,98.35,98.27,98.58,98.55,98.48,98.59,98.49,98.55,98.62,
                98.51,98.62,98.15,98.42,97.04,98.31,98.58,98.33,98.41,98.31,98.10,98.27,98.36,98.31,98.53])
# c_g_acc_ve4 v矩阵*1e-4
c_g_acc_ve4=  np.array([97.39,98.78,98.69,98.47,98.28,98.48,98.52,98.34,98.43,98.39,98.43,98.60,98.05,98.58,98.65,
               98.57,98.14,97.94,98.57,98.93,98.55,99.01,98.49,98.08,98.15,98.41,98.50,98.36,98.24,97.88])
# c_g_v_e4为v矩阵*1e-4的最后一次迭代
c_g_v_e4=    np.array([98.45,97.82,98.57,98.56,98.35,98.61,98.34,98.01,98.59,98.39,98.55,98.09,97.98,98.19,98.43,
                98.19,98.30,98.71,97.95,98.40,98.48,98.85,98.46,98.57,98.65,98.66,98.51,98.42,98.28,98.65])


# #BoxPlot for Fig_8 (average accuracy)
label='acc','l','mse','l_mse','ve4','l_ve4'
fig1, ax = plt.subplots()
plt.figure(1)
ax.boxplot([c_g_acc,c_g,c_g_acc_mse,c_g_mse,c_g_acc_ve4,c_g_v_e4],
            notch=False, labels = label,patch_artist = False, medianprops={'color':'red'},boxprops = {'color':'blue','linewidth':'1.0'},
            capprops={'color':'black','linewidth':'1.0'})
color = ['blue', 'orange', 'green','red','purple']  # 有多少box就对应设置多少颜色
# ax.legend(['RF-GAN'], loc='upper right')
# plt.ylim(ymin=0,ymax=100)
# plt.savefig('../plot/boxplot_b.png')
plt.show()
#
