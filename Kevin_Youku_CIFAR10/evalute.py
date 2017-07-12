# 对训练好的模型进行 准确率验证

import tensorflow as tf
import numpy as np
import cifar10_input
import cifar10
import math
import os
import os.path

# BATCH_SIZE = 128
BATCH_SIZE = 128

def evaluate():
    with tf.Graph().as_default():

        log_dir = 'log_2/'
        test_dir = 'data/'

        n_test = 10000

        # read test data
        images, labels = cifar10_input.read_cifar10(data_dir=test_dir,is_train=False, batch_size=BATCH_SIZE,is_suffle=False)

        logits = cifar10.inference(images)

        tot_k_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            print('reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading success, global_step is %s' % global_step)
            else:
                print('no checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE)) # 取整数
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and  not coord.should_stop():
                    predictions = sess.run([tot_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision )

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()

            coord.join(threads)


# evaluate()
# 验证一张图的分类及概率
def evalute_one_image_():
    with tf.Graph().as_default():

        log_dir = 'log_2/'
        test_dir = 'data/'

        n_test = 10000

        # read test data
        images, labels = cifar10_input.get_one_image()

        logits = cifar10.inference(images)

        tot_k_op = tf.nn.in_top_k(logits, labels, 1)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            print('reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading success, global_step is %s' % global_step)
            else:
                print('no checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE)) # 取整数
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and  not coord.should_stop():
                    predictions = sess.run([tot_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    precision = true_count / total_sample_count
                print('precision = %.3f' % precision )

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()

            coord.join(threads)


# 验证一张图的分类及概率
def evalute_one_image():

   train_dir = 'data/train/'
   train, train_label = input_data.get_file(train_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1,208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, [208, 208, 3])

       logs_train_dir = 'logs/train'
       saver = tf.train.Saver()

       with tf.Session() as sess:
           print('Reading checkpoints...')
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:

              # 拿到训练好的模型
              global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
              saver.restore(sess, ckpt.model_checkpoint_path)

              print('loading success, global_step is %s' % global_step)
           else:
               print('no checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index == 0:
               print('this is a cat with possibility %.6f' %prediction[:,0])
           else:
               print('this is a dog with possibility %.6f' %prediction[:,1])


# 测试集中随机抽取 一张图片
def get_one_image():
    pass

# evalute_one_image_()