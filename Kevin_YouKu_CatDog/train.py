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
MAX_STEP = 4000  #建议>10kR
learning_rate = 0.02 #建议< 0.0001

def run_training():
    train_dir = 'data/train/'
    logs_train_dir = 'logs/train/'
    logs_val_dir = 'logs/val'

    train, train_labels, val, val_labels = input_data.get_file(train_dir, 0.2) # train 为ndarray图片路径
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_labels,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val,
                                                      val_labels,
                                                      IMG_W,
                                                      IMG_H,
                                                      BATCH_SIZE,
                                                      CAPACITY)
   # 原来的
    '''
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
    '''
    # 现在边训练边验证
    logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, train_label_batch)
    train_op = model.training(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    # 在训练过程中 不断送进图片和label
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE])

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 将训练数据 和验证数据同时写入到tb文件
        summary_op = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x:tra_images, y_:tra_labels}
                                                )
                if step % 50 == 0:
                    print('step: %d, train loss: %.2f, train accuracy= %.2f%%' %(step, tra_loss, tra_acc*100))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                # 每间隔 200步 验证一下
                if step % 200 == 0:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x:val_images, y_:val_labels }
                                                 )
                    print('**step %d, val loss: %.3f val accuracy: %.3f ** '%(step, val_loss, val_acc*100))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('done training - epoch limit reachea')

        finally:
            coord.request_stop()

        coord.join(threads)










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



















