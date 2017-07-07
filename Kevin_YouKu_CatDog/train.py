# coding=utf-8

# 训练
import os
import tensorflow as tf
import numpy as np
import input_data
import model

# 一些参数
N_CLASSES = 2
IMG_H = 208
IMG_W = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000  #建议>10kR
learning_rate = 0.0001 #建议< 0.0001

def run_training():
    train_dir = 'data/train/'
    logs_train_dir = 'logs/train/'

    image, train_labels = input_data.get_file(train_dir)
    train_batch, train_label_batch = input_data.get_batch(image,
                                                          train_labels,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = model.inference(train_batch,BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, train_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('step %d ,train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, train_acc*100.0))
                summary_str = sess.run(summary_op)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done train -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()



run_training()


# # 用训练好的模型 验证一张图片是猫是狗
# from  PIL import Image
# import matplotlib.pyplot as plt
#
# # 从训练图片中随机取一张图片
# def get_one_image(train):
#
#
#
#     n = len(train)
#     ind = np.random.randint(0, n)
#     img_dir = train[ind]
#
#     image = Image.open(img_dir)
#     plt.imshow(image)
#     image = image.resize([208, 208])
#     image = np.array(image)
#     return image
#
#
# # 验证
# def evaluate_one_image():
#
#    train_dir = 'data/train/'
#    train, train_label = input_data.get_file(train_dir)
#    image_array = get_one_image(train)
#
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 2
#
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1,208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#
#        logit = tf.nn.softmax(logit)
#
#        x = tf.placeholder(tf.float32, [208, 208, 3])
#
#        logs_train_dir = 'logs/train'
#        saver = tf.train.Saver()
#
#        with tf.Session() as sess:
#            print('Reading checkpoints...')
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#               saver.restore(sess, ckpt.model_checkpoint_path)
#
#               print('loading success, global_step is %s' % global_step)
#            else:
#                print('no checkpoint file found')
#
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            max_index = np.argmax(prediction)
#            if max_index == 0:
#                print('this is a cat with possibility %.6f' %prediction[:,0])
#            else:
#                print('this is a dog with possibility %.6f' %prediction[:,1])
#
#
# evaluate_one_image()



















