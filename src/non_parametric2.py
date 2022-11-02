# 支持向量机
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon

def boxplot(data_list,label,dir):
    fig, ax = plt.subplots()
    ax.boxplot(data_list,
                notch=False, labels = label,patch_artist = False, medianprops={'color':'red'},boxprops = {'color':'blue','linewidth':'1.0'},
                capprops={'color':'black','linewidth':'1.0'})
    color = ['blue', 'orange', 'green','red','purple']  # 有多少box就对应设置多少颜色
    # ax.legend(['RF-GAN'], loc='upper right')
    # plt.ylim(ymin=0,ymax=100)
    fig.savefig(dir)
    plt.show()

def post_hoc(x1,x2):
    stat_post, p_post = wilcoxon(x1,x2)
    print('wilcoxon Post hoc: ',stat_post, p_post)
    # interpret
    alpha = 0.05
    if p_post > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

#以下为非batchnorm 操作后的生成数据
# normal (with acc metric)
c_g_acc= np.array(    [92.00,91.99,91.93,91.06,92.17,92.65,91.65,91.46,91.44,90.97,91.75,90.91,90.31,92.11,91.40,
               91.76,92.56,91.63,92.68,91.13,91.32,91.77,91.90,91.97,91.32,92.17,92.41,92.86,91.93,91.47])
# normal_l (last model)
c_g=     np.array(      [90.05,90.93,91.72,90.97,92.60,91.36,91.08,91.16,91.52,92.61,92.62,90.86,92.47,92.52,92.09,
                90.83,91.45,91.27,91.75,90.97,91.34,90.50,91.39,91.89,90.98,91.79,92.07,91.99,90.66,91.06])

# mse (with acc metric)
c_g_acc_mse= np.array( [92.31,91.40,90.65,91.79,89.81,91.95,92.44,92.34,92.46,91.04,91.50,93.00,91.10,92.36,93.33,
               92.18,90.93,91.73,91.87,92.40,92.92,92.02,91.78,91.65,92.53,92.83,91.73,93.27,91.32,92.20])
# mse_l (last model)
c_g_mse=     np.array(  [92.51,92.95,92.25,92.50,92.52,92.04,92.50,91.55,92.84,93.10,92.75,91.92,91.89,92.52,93.12,
                92.04,92.77,92.37,93.11,92.70,92.83,91.67,92.39,92.91,92.41,92.13,92.55,91.72,93.14,92.48])

# # c_g_acc_v为wgan+vapnik 损失
# c_g_acc_v=   np.array( [92.75,93.77,93.15,93.21,92.73,92.59,93.90,93.27,92.41,92.34,93.51,93.41,92.34,92.82,93.48,
#                92.60,93.70,93.12,93.34,93.24,91.59,93.42,93.20,93.24,92.53,92.16,91.50,92.93,92.95,92.56])
# # c_g_v为wgan+vapnik损失的最后一次迭代
# c_g_v=      np.array(  [91.98,91.34,92.86,92.57,91.72,92.69,92.39,91.92,91.91,92.05,93.22,92.30,92.27,92.32,93.30,
#                91.66,93.14,91.73,92.20,92.79,91.87,92.36,92.69,92.76,92.76,90.67,92.77,93.15,92.35,93.20])

# v (with acc metric)
c_g_acc_ve4=  np.array([94.00,93.71,94.27,93.47,93.33,94.61,93.90,93.75,93.70,94.86,94.23,93.65,94.67,94.68,93.63,
               94.05,95.10,93.26,93.41,95.21,93.71,94.04,94.29,94.22,94.09,95.17,93.76,94.16,94.23,94.13])
# v_l (last model)
c_g_v_e4=    np.array([93.95,93.32,93.78,93.61,93.76,93.10,93.58,93.05,93.55,94.20,94.16,93.78,93.82,94.14,93.57,
                94.13,93.47,93.96,93.50,95.00,93.67,93.02,94.39,94.52,93.88,94.41,94.36,93.75,94.04,92.85])
# smote
smote=       np.array(  [91.94,91.93,91.94,91.95,91.94,91.94,91.95,91.94,91.94,91.95,91.94,91.95,91.94,91.95,91.94,
                91.95,91.95,91.94,91.94,91.94,91.95,91.95,91.93,91.94,91.94,91.93,91.94,91.95,91.94,91.94])
# imbalance
i=         np.array(    [56.75]*30)

# diego's idea (model selection)
RF_GAN=      np.array(  [91.05,90.52,91.06,90.59,89.71,89.28,90.75,90.03,89.78,90.32,85.38,89.47,90.82,88.84,84.16,
                91.74,82.64,88.27,90.89,89.84,89.49,89.41,89.80,91.27,90.10,91.14,90.22,91.24,86.14,84.83])

# unsupervised GAN (last model)
RF_GAN2=     np.array(  [81.39,88.33,82.87,88.75,89.62,88.97,90.50,83.70,91.68,89.07,76.95,87.72,90.35,90.15,84.00,
                90.84,87.99,89.33,90.36,90.57,86.89,90.60,82.00,90.72,90.38,89.81,90.85,90.71,88.95,81.00])


# def ModelSensity test
stat, p = friedmanchisquare(c_g_acc,c_g,c_g_acc_mse,c_g_mse,c_g_acc_ve4,c_g_v_e4,i,smote,RF_GAN,RF_GAN2)
print('Friedman Test: ',stat, p)
alpha = 0.05
if p > alpha:
     print('There is no difference between them (fail to reject H0)')
else:
    print('There is difference between them (reject H0) and Jump to the Post hoc!')
    print('*'*52)
    print('-'*20+'pos hoc test'+'-'*20)
    post_hoc(c_g_acc_ve4,c_g_v_e4)
    print('*' * 52)


# # #BoxPlot for Fig_8 (average accuracy)
plt.figure(1)
save_path='./fig/boxplot.png'
labels='acc','l','mse','l_mse','ve4','l_ve4'
data_lists=[c_g_acc,c_g,c_g_acc_mse,c_g_mse,c_g_acc_ve4,c_g_v_e4]
boxplot(data_list=data_lists,label=labels,dir=save_path)


# 数据不平衡
# plt.figure(2)
# save_path='../plot/boxplot2.png'
# labels='normal','normal_l','mse','mse_l','v','v_l','i','smote','rf_gan','rf_gan2'
# data_lists=[c_g_acc,c_g,c_g_acc_mse,c_g_mse,c_g_acc_ve4,c_g_v_e4,i,smote,RF_GAN,RF_GAN2]
# boxplot(data_list=data_lists,label=labels,dir=save_path)

# #不同随机数 （ve4）
# plt.figure(3)
# label=['acc','last']
# # plt.legend(label,loc=0,ncol=2)
# plt.plot(c_g_acc_ve4,'bo-')
# plt.plot(c_g_v_e4,'ro-')
# plt.ylabel('Different random seeds')
# # plt.legend(label,loc=1,ncol=1)
# plt.savefig('../plot/different_seeds.png')
# plt.show()
#
# # plot confusion matrix with best
# name_list=['normal_l','normal_acc','mse_l','mse_acc','ve4_l','ve4_acc']
# i=0
# for name in name_list:
#     cf=pd.read_csv('../plot/cf_33333/'+name+'.csv')
#     conf_matrix=cf.iloc[0:4,1:5]
#     import seaborn as sn
#     plt.figure(i)
#     print(i)
#     i+=1
#     ax= plt.subplot()
#     list=['a','b','c','d']
#     df_cm = pd.DataFrame(conf_matrix._values/6000,
#                          index = [i for i in list],
#                          columns = [i for i in list])
#     sn.heatmap(df_cm, annot=True, cmap='gist_gray_r',fmt=".2f",cbar=False)
#     ax.set_title('cf_'+name)
#     plt.savefig('../plot/'+name+'.png')
#     plt.show()
#
#
