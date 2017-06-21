
#安装
pip install tensorboard

#用命令行启动
tensorboard --logdir=/tmp/tensorflow/mnist/logs/fully_connected_feed

# /tmp/tensorflow/mnist/logs/fully_connected_feed 为mnist训练过程中写入的即时数据的 路径


# 训练过程写入即时数据
summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

# 在浏览器中打开 http://localhost:6006/