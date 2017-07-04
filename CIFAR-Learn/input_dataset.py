# coding=utf-8
# 参考 http://blog.csdn.net/diligent_321/article/details/53130913

# 数据输入部分

import os
import tensorflow as tf

# 原图像的尺寸为32*32 ，但是根据常识，信息部分一般位于图像中央，这里定义裁剪后的图像尺寸
fixed_height = 24
fixed_width = 24
# cifar数据集格式，训练例集和测试例集分别为 50k和 10k
train_samples_per_epoch = 50000
test_sample_per_epoch = 10000
data_dir = './cifar-10-batches-bin' # 数据集所在文件夹
batch_size = 128

def read_cifar10(filename_queue):

    # 定义一个空的类对象
    class Image(object):
        pass
    image = Image()
    image.height = 32
    image.width = 32
    image.depth = 3
    label_bytes = 1
    image_bytes = image.height * image.width * image.depth
    Bytes_to_read = label_bytes + image_bytes

    # 定义一个Reader， 它每次能从文件中读取固定字节数
    reader = tf.FixedLengthRecordReader(record_bytes=Bytes_to_read)
    # 返回从filename_queue 中读取的（key， value）对，key，value都是字符串型的tensor，
    # 并且当队列中的某一个文件读完成后，该文件会dequeue
    image.key, value_str = reader.read(filename_queue)
    # 解码操作可以看作读二进制文件，把字符串中的字节转换为数值向量,每一个数值占用一个字节
    # 在[0, 255]区间内，因此out_type要取uint8类型
    value = tf.decode_raw(bytes=value_str, out_type=tf.uint8)
    # 从一维tensor对象中截取一个slice，类似从一维向量中筛选子向量，因为value中包含了label和feature
    # 故需要对tensor进行 'parse'操作
    # begin 和 size待截取片段的起点和长度
    image.label = tf.slice(input_=value, begin=[0], size=[label_bytes])
    data_mat = tf.slice(input_=value, begin=[label_bytes], size=[image_bytes])
    # 这里的维度顺序是根据cifar二进制文件格式而定的
    data_mat = tf.reshape(data_mat, (image.depth, image.height, image.width))
    # 对data_mat的维度进行重新排序，返回值的第i个维度对应着data_mat的第perm[i]
    transposed_value = tf.transpose(data_mat,perm=[1, 2, 0])
    image.mat = transposed_value
    return image

def get_batch_samples(img_obj, min_samples_in_queue, batch_size, shuffle_flag):
    '''''
    tf.train.shuffle_batch()函数用于随机的shuffling队列中的tensor来创建batches\
    (也即每次可以读取多个data文件中的样例构成一个batch) 这个函数向当前Graph中添加了下列对象：
    * 创建了一个shuffling queue， 用于把tensors中的tensor是压入该队列
    * 一个 dequeue_many操作， 用于根据队列中的数据创建一个batch
    * 创建了一个ueueRunner对象， 用于启动一个进程压数据到队列
    capacity参数用于控制shuffling queue的最大长度， min_after_dequeue参数表示进行一次dequeue操作后队列中元素的最小数量，
    可以用于确保batch中元素的随机性，num_threads参数用于指定多少个threads负责压tensor到队列；enqueue_many参数用于表征是否tensor中
    的每一个tensor都代表一个样例，tf.train.batch()与之类似，只不过顺序的出队列（每次只能从一个data文件中读取datch），少了随机性
    '''''
    if shuffle_flag == False:
        image_batch, label_batch = tf.train.batch(
            tensors=img_obj,
            batch_size=batch_size,
            num_threads=4,
            capacity=min_samples_in_queue + 3*batch_size
        )
    else:
        image_batch, label_batch = tf.train.shuffle_batch(
            tensors=img_obj,
            batch_size=batch_size,
            num_threads=4,
            capacity=min_samples_in_queue + 3 * batch_size
        )

    # 输出预处理后的图像的summary缓存对象， 用于在session中写入到事件文件中
    # tf.image_summary('input_iamge', image_batch, max_images=6)
    tf.summary.image('input_iamge', image_batch, max_outputs=6)

    return image_batch, tf.reshape(label_batch, shape=[batch_size])

    pass

def preprocess_input_data():
    # 这部分用于对训练数据进行'数据增强'操作，通过增加训练集的大小来防止过拟合
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    # filename = [os.patch.join(data_dir, 'test_batch.bin')]
    for f in filenames: # 检验训练数据集文件是否存在
        if not tf.gfile.Exists(f):
            raise ValueError('fail to find file --' +f)

    # 把文件名输入到队列中，作为整个data pipe的第一阶段
    filename_queue = tf.train.string_input_producer(string_tensor=filenames)
    # 从文件名队列中读取一个tensor类型的图像
    image = read_cifar10(filename_queue)
    new_img = tf.cast(image.mat, tf.float32)
    # 输出预处理前图像的summary缓存对象
    tf.image_summary('raw_input_image', tf.reshape(new_img, [1, 32, 32, 3]))
    # 从原图中切出子图像
    new_img = tf.random_crop(new_img,size= (fixed_height, fixed_widht, 3))
    # 随机调节图像亮度
    new_img = tf.image.random_brightness(new_img, max_delta=63)
    # 随机左右翻转图像
    new_img = tf.image.random_flip_left_right(new_img)
    # 随机调整图像对比度
    new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8)
    # 对图像进行whiten操作，目的是降低图像的冗余性， 尽量去除输入特征间的相关性
    final_img = tf.image.per_image_whitening(new_img)

    # 用于确保读取到的batch中的样例的随机性， 使其覆盖到更多的类别， 更多的数据文件
    min_samples_ratio_in_queue = 0.4
    min_samples_in_queue = int(min_samples_ratio_in_queue * train_samples_per_epoch)
    return get_batch_samples([final_img, image.label], min_samples_in_queue, batch_size, shuffle_flag=True)






































