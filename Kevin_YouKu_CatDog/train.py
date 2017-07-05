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
MAX_STEP = 10000  #建议>10k
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






















