import tensorflow as tf
import numpy as np
import os
import os.path
import input_data
import tools
import VGG

IMG_W = 32
IMG_H = 32
N_CLASS = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000
IS_PRETRAIN = False

# 训练
def train():

    pre_trained_weights = 'vgg16_pretrain/vgg16.npy'
    # tools.test_load()

    data_dir = 'data'
    train_log_dir = 'logs/train'
    val_log_dir = 'logs/val'

    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir,
                                                                   is_train=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   is_suffle=True)
        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=False,
                                                                   batch_size=BATCH_SIZE,
                                                                   is_suffle=False)


    my_global_step = tf.Variable(0,name='global_step', trainable=True)

    logits = VGG.VGG16N(tra_image_batch, N_CLASS, IS_PRETRAIN)
    loss = tools.loss(logits, tra_label_batch)
    accuracy = tools.accuracy(logits,tra_label_batch)
    train_op = tools.optimize(loss, learning_rate, my_global_step)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASS])


    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    # tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={
                                                x:tra_images,
                                                y_:tra_labels
                                            })

            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('step %d, loss %.4f accuracy %4f' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
                # 打印一下所有的变量
                # tools.print_all_variables(False)

            if step % 100 == 0 or (step + 1) == MAX_STEP:  # 验证一下
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={
                                                 x:val_images,
                                                 y_:val_labels
                                             })
                print('** step %d ,val loss=%.2f, val accracy=%.2f%%  **' %(step, val_loss, val_acc))
                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


train()



# 测试数据集中验证模型
import math
def evaluate():
    with tf.Graph().as_default():
        log_dir = 'logs2_test'
        test_dir = 'data'
        n_test = 10000

        images, labels = input_data.read_cifar10(test_dir,
                                                 is_train=False,
                                                 batch_size=BATCH_SIZE,
                                                 is_suffle  =False)

        logits = VGG.VGG16(images, N_CLASS, IS_PRETRAIN)

        correct = tools.num_correct_predition(logits, labels)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            print('reading checkpoint')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)

                print('loading success,global_step is %s' %global_step)

            else:
                print('no checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluting...')
                num_iter = int(math.ceil(n_test/BATCH_SIZE)) # 取整
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    preditions = sess.run(correct)
                    true_count += np.sum(preditions)
                    step +=1
                    precisioin = true_count / total_sample_count

                print('precision = %.3f%%' %(precisioin*100.0))

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                # coord.join(threads)

            coord.join(threads)





















































