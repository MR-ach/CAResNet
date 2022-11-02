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
cnn = np.array([0.879333333,    0.886191667,    0.897508333,
                0.888466667,    0.885725,       0.903225,
                0.870316667,    0.875258333,    0.894058333,
                0.8995,	        0.865416667,	0.892508333,
                0.899,	        0.872783333,	0.88325,
                0.869533333,	0.880533333,	0.877508333,
                0.865383333,	0.873633333,	0.888525,
                0.885216667,	0.886916667,	0.905675,
                0.881158333,	0.881683333,	0.876416667,
                0.873175,	    0.883733333,	0.883766667
])

cnn_eca = np.array([0.878183333,	0.874908333,	0.892433333,
                    0.889908333,	0.889641667,	0.889666667,
                    0.871966667,	0.882058333,	0.891825,
                    0.874333333,	0.869166667,	0.877275,
                    0.863316667,	0.880891667,	0.886533333,
                    0.8886,	        0.88575,	    0.883625,
                    0.888683333,	0.885458333,	0.869308333,
                    0.884141667,	0.88645,	    0.884016667,
                    0.885675,	    0.88255,	    0.89035,
                    0.864266667,	0.879191667,	0.879933333
])

cnn_resnet = np.array([0.900583333,	0.887691667,	0.914375,
                       0.897975,	0.886891667,	0.879183333,
                       0.925125,	0.932108333,	0.883075,
                       0.910291667,	0.8936,	        0.874483333,
                       0.901491667,	0.908341667,	0.912033333,
                       0.895916667,	0.916566667,	0.872383333,
                       0.910708333,	0.890866667,	0.892891667,
                       0.896141667,	0.890033333,	0.90405,
                       0.873508333,	0.905908333,	0.912816667,
                       0.927083333,	0.914108333,	0.875291667
])

cnn_res_eca = np.array([0.92615,	    0.916441667,	0.922283333,
                        0.912166667,	0.888091667,	0.910633333,
                        0.918533333,	0.929333333,	0.941475,
                        0.906966667,	0.912358333,	0.931808333,
                        0.9371,	        0.904133333,	0.928833333,
                        0.894875,	    0.910191667,	0.872675,
                        0.9037,	        0.904975,	    0.932841667,
                        0.944675,	    0.908383333,	0.904833333,
                        0.887958333,	0.914775,	    0.913358333,
                        0.892508333,	0.922225,	    0.917766667
])

# def ModelSensity test
stat, p = friedmanchisquare(cnn,cnn_eca,cnn_resnet,cnn_res_eca)
print('Friedman Test: ',stat, p)
alpha = 0.05
if p > alpha:
     print('There is no difference between them (fail to reject H0)')
else:
    print('There is difference between them (reject H0) and Jump to the Post hoc!')
    print('*'*52)
    print('-'*20+'pos hoc test'+'-'*20)
    post_hoc(cnn,cnn_eca)
    print('*' * 52)

# mean = np.mean(cnn_res_eca)
# print(mean)

# #BoxPlot for Fig_8 (average accuracy)
# plt.figure(1)
save_path='./fig/boxplot.png'
labels='1DCNN','1DECA','1DRNN','MSA-1DCNN'
data_lists=[cnn,cnn_eca,cnn_resnet,cnn_res_eca]

boxplot(data_list=data_lists,label=labels,dir=save_path)
plt.ylim(ymin=0,ymax=1)
plt.savefig('./fig/boxplot.png')
plt.show()