# coding=utf-8
# 参考http://blog.csdn.net/han____shuai/article/details/69939623



import numpy as np
import struct
import matplotlib.pyplot as plt

def loadImageSet(filename):
    print 'load image set', filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print 'head: ', head

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]

    #[6000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width*height])
    print 'load imgs finished'
    return imgs

def loadLabelSet(filename):
    print 'load label set', filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print 'head', head
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + 'B'
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print 'load label fiished'
    return  labels


# 测试是否成功
def show_one_img():
    filename = 'MNIST_data/train-images-idx3-ubyte'
    binfile = open(filename, 'rb')
    buf = binfile.read()

    index = 0
    magic, numImages, numRows, numColums = struct.unpack_from('>IIII', buf, index )
    index += struct.calcsize('>IIII')

    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')

    print 'im--1: ', im
    im = np.array(im)
    print 'im--2: ', im
    im = np.reshape(28, 28)
    print 'im--1: ', im



    fig = plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    imgs = loadImageSet('MNIST_data/train-images-idx3-ubyte')
    labels = loadLabelSet('MNIST_data/train-labels-idx1-ubyte')

    show_one_img()
    pass

    for i in range(10):
       print i


    for i in range(3):
        one_img = imgs[i]

        print one_img

        plt.imshow(one_img)
        plt.show()

'''
import numpy as np
import struct


def loadImageSet(filename):
    print "load image set", filename
    # open可用相对路径，也可用绝对路径
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print "head,", head

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print "load imgs finished"
    return imgs


def loadLabelSet(filename):
    print "load label set", filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print "head,", head
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print 'load label finished'
    return labels


if __name__ == "__main__":
    imgs = loadImageSet("/Users/mac/Tensorflow-Learn/tensorflow 学习/MNIST_data/train-images-idx3-ubyte")
    labels = loadLabelSet("/Users/mac/Tensorflow-Learn/tensorflow 学习/train-labels-idx1-ubyte")
    
'''













