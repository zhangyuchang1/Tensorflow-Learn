
# coding=utf-8

# path = "/Users/mac/Tensorflow-Learn/tensorflow 学习/MNIST_data/train-images-idx3-ubyte"
path = "MNIST_data/train-images-idx3-ubyte"

binfile = open(path, 'rb')
buffers = binfile.read()

print (binfile)
print (buffers)