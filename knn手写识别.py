import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import random
from six.moves import urllib
from datetime import datetime

#按32位读取，主要为读校验码、图片数量、尺寸准备的
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
def read_image(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print(magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows * cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data

#抽取标签，导入测试集
def read_label(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

#KNN算法
def KNN(test_data, images, labels, k):
    dataSetSize = images.shape[0]
    # np.tile(A,B)：重复A B次，相当于重复[A]*B
    # print np.tile(newInput, (numSamples, 1)).shape
    distance1 = np.tile(test_data, (dataSetSize)).reshape((dataSetSize,784))-images
    distance2 = distance1**2   #计算欧式距离
    distance3 = distance2.sum(axis=1)
    distances4 = distance3**0.5
    sortedDistIndicies = distances4.argsort()  #排序
    classCount = np.zeros((10), np.int32)
    for i in range(k):  # 选取前K个
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] += 1
    return np.argmax(classCount), sortedDistIndicies[0:k]

#计算预测失败率
def KNN_ACCURACY(train_x,test_x,train_y,test_y,k):
    testNum = test_x.shape[0]
    print('测试图片数量:',testNum)
    errorCount = 0
    for i in range(testNum):
        result,sortlist = KNN(test_x[i], train_x, train_y, k)
        if result != test_y[i]:
            errorCount += 1.0 #不一样图像数目
    error_rate = errorCount / float(testNum)  #误分率
    return error_rate

#输出相似的K个图像
def KNN_PRINT(train_x,test_x,train_y,test_y,test_id,k):
    result,sortlist = KNN(test_x[test_id], train_x, train_y, k)
    print('预测结果为:',result,'实际结果为:',test_y[test_id])
    fig=plt.figure(figsize=(8,8))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    for i in range(len(sortlist)):
        images = np.reshape(train_x[sortlist[i]], [28,28])
        ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
        ax.imshow(images,cmap=plt.cm.binary,interpolation='nearest')
        ax.text(0,7,str(train_y[sortlist[i]]))
    plt.show()
    plt.pause(5)
    plt.close(fig)

# 主函数，先读图片，然后用于测试手写数字
def testHandWritingClass():
    train_x = read_image('mnist/train_images', True, True)
    train_y = read_label('mnist/train_labels')
    test_x = read_image('mnist/test_images', True, True)
    test_y = read_label('mnist/test_labels')
    trainNum = 7000  # 训练图片数
    testNum = 1000  # 测试图片数
    k = 25  # 邻近图片数目
    train_Start = random.randint(0, 60001 - trainNum)
    train_End = train_Start + trainNum
    train_i = train_x[train_Start:train_End, :]
    train_l = train_y[train_Start:train_End]
    test_Start = random.randint(0, 10001 - testNum)
    test_End = test_Start + testNum
    test_i = test_x[test_Start:test_End, :]
    test_l = test_y[test_Start:test_End]

    # 输出K-misclassification rate曲线
    import matplotlib
    matplotlib.use('TkAgg')
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    trainNum = 7000  # 训练图片数
    testNum = 1000 # 测试图片数

    x = np.arange(1, 10, 1)
    y = []
    for t in x:
        print('\n第 ', t, ' 次测试')
        ## step 1: load data
        print("step 1: load data...")
        # 输出K个邻近图像
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        test_id = random.randint(0, 10000)
        KNN_PRINT(train_i, test_x, train_l, test_y, test_id, k)
        ## step 2: training...
        print("step 2: training...")
        pass
        ## step 3: testing...
        print("step 3: testing...")
        train_Start = random.randint(0, 60001 - trainNum)  # 随机选择
        train_End = train_Start + trainNum
        train_i = train_x[train_Start:train_End, :]
        train_l = train_y[train_Start:train_End]
        test_Start = random.randint(0, 10001 - testNum)
        test_End = test_Start + testNum
        test_i = test_x[test_Start:test_End, :]
        test_l = test_y[test_Start:test_End]
        a = datetime.now()
        for i in range(testNum):
            KNN(test_x[i], train_x, train_y, t)
            if i % 100 == 0:
                print("完成%d张图片" % (i))
        b = datetime.now()
        print('训练集图片数量：', train_i.shape[0], '区间:', train_Start, '~', train_End)
        print('测试集图片数量：', test_i.shape[0], '区间:', test_Start, '~', test_End)
        misclassification_rate = KNN_ACCURACY(train_i, test_i, train_l, test_l, t)
        ## step 4: show the result
        print("step 4: show the result...")
        print("一共运行了%d秒" % ((b - a).seconds))
        print('K=', t, 'accuracy: %.2f%%' % ((1 - misclassification_rate) * 100))
        y.append(misclassification_rate)
    plt.plot(x, y, label='KNN')  # 曲线格式
    plt.xlabel("K")
    plt.ylabel("misclassification rate")
    plt.ylim(0, 2 * max(y))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    testHandWritingClass()
