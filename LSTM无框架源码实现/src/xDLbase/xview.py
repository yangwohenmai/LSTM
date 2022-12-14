"""
Function For Show Result
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 训练完成后，生成训练过程图像
# 测试和训练的acc和loss线，测试和验证对比
class ResultView(object):
    '''
     (Params.EPOCH_NUM,
      ['train_loss', 'val_loss', 'train_acc', 'val_acc'],
      ['k', 'r', 'g', 'b'],
      ['Iteration', 'Loss', 'Accuracy'],
      Params.DTYPE_DEFAULT)
    '''

    def __init__(self, epoch, line_labels, colors, ax_labels, dataType):
        self.cur_p_idx = 0
        self.curv_x = np.zeros(epoch * 100, dtype=int)
        self.curv_ys = np.zeros((4, epoch * 100), dtype=dataType)
        self.line_labels = line_labels
        self.colors = colors
        self.ax_labels = ax_labels


    def addData(self, curv_x, loss, loss_v, acc, acc_v):

        self.curv_x[self.cur_p_idx] = curv_x
        self.curv_ys[0][self.cur_p_idx] = loss
        self.curv_ys[1][self.cur_p_idx] = loss_v
        self.curv_ys[2][self.cur_p_idx] = acc
        self.curv_ys[3][self.cur_p_idx] = acc_v
        self.cur_p_idx += 1

    # 显示曲线
    def show(self):
        self.showCurves(self.cur_p_idx, self.curv_x, self.curv_ys, self.line_labels, self.colors, self.ax_labels)

    def showCurves(self, idx, x, ys, line_labels, colors, ax_labels):
        lsArr = [':','-']
        LINEWIDTH = 2.0
        plt.figure(figsize=(8, 4))
        # loss
        ax1 = plt.subplot(211)
        for i in range(2):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], ls=lsArr[i%2],linewidth=LINEWIDTH, label=line_labels[i])

        ax1.xaxis.set_major_locator(MultipleLocator(4000))
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax1.set_xlabel(ax_labels[0])
        ax1.set_ylabel(ax_labels[1])
        plt.grid()
        plt.legend()

        # Acc
        ax2 = plt.subplot(212)
        for i in range(2, 4):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], ls=lsArr[i%2], linewidth=LINEWIDTH, label=line_labels[i])

        ax2.xaxis.set_major_locator(MultipleLocator(4000))
        ax2.yaxis.set_major_locator(MultipleLocator(0.02))
        ax2.set_xlabel(ax_labels[0])
        ax2.set_ylabel(ax_labels[2])

        plt.grid()
        plt.legend()
        plt.show()
        plt.close()

    @staticmethod
    def v2img(v):
        # 展示n组mnist图片，每组10个
        i =0
        n = v.shape[0]
        v1 = v.reshape(n,10,28,28)
        n = v1.shape[0]
        fig = plt.figure()
        for row in range(n):
            for col in range(10):
                i=i+1
                plotwindow = fig.add_subplot(n,10,i)
                plt.axis('off')
                plt.imshow(v1[row][col], cmap='gray')
        plt.show()
        # plt.savefig("test.png")  # 保存成文件
        plt.close()
# 训练完成后，生成训练过程图像
# 生成多条acc和loss线的对比线，验证单独一张图片，测试需要的话放另外一张图片
class ResultViewM(object):
    '''
     (Params.EPOCH_NUM,
      ['SGD', 'Momentum', 'NAG', 'Adagrad','RMSprop','AdaDelta','Adam'],
      ['k','y', 'r', 'g', 'b','m','c'],
      ['Iteration', 'Loss', 'Accuracy'],
      Params.DTYPE_DEFAULT)
    '''

    def __init__(self, epoch, line_labels, colors, ax_labels, dataType):
        self.cur_p_idx = 0 # 每个不同优化方法，要把cur_p_idx 置为0
        self.curv_x = np.zeros(epoch * 100, dtype=int)
        self.line_labels = line_labels
        self.colors = colors
        self.curv_ys = np.zeros((len(line_labels)*2, epoch * 100), dtype=dataType)#数据标签
        self.ax_labels = ax_labels # 坐标轴标签

    def idxadd(self):
        self.cur_p_idx += 1

    def addData(self, curv_x, loss, acc, y_seq):

        self.curv_x[self.cur_p_idx] = curv_x # x值
        self.curv_ys[y_seq][self.cur_p_idx] = loss # y_seq：SDG-loss:0，Adam-loss:6;SDG-acc:7
        self.curv_ys[y_seq+len(self.line_labels)][self.cur_p_idx] = acc
        # self.cur_p_idx += 1

    # 显示曲线
    def show(self):
        self.showCurves(self.cur_p_idx, self.curv_x, self.curv_ys, self.line_labels, self.colors, self.ax_labels)

    def showCurves(self, idx, x, ys, line_labels, colors, ax_labels):
        LINEWIDTH = 1.0 # 线宽
        plt.figure(figsize=(8, 4))
        # loss
        ax1 = plt.subplot(121) # 上下窗的上图
        for i in range(len(self.line_labels)):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax1.xaxis.set_major_locator(MultipleLocator(300)) # 横坐标步长
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))  # 纵坐标步长
        ax1.set_xlabel(ax_labels[0])
        ax1.set_ylabel(ax_labels[1])
        plt.grid()
        plt.legend()

        # Acc
        ax2 = plt.subplot(122) # 上下窗的下图
        for i in range(len(self.line_labels), len(self.line_labels) * 2):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i-len(self.line_labels)], linewidth=LINEWIDTH, label=line_labels[i-len(self.line_labels)])

        ax2.xaxis.set_major_locator(MultipleLocator(300)) # 横坐标步长
        ax2.yaxis.set_major_locator(MultipleLocator(0.05)) # 纵坐标步长
        ax2.set_xlabel(ax_labels[0])
        ax2.set_ylabel(ax_labels[2])

        plt.grid()
        plt.legend()
        plt.show()
