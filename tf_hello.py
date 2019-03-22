#!/usr/bin/python3

import tensorflow as tf
#tf.enable_eager_execution()
#tf.add(1, 2).numpy()

hello = tf.constant('Hello, TensorFlow!')
#hello.numpy()

sess = tf.Session()
print(sess.run(hello))

print("TF built with cuda: " +str(tf.test.is_built_with_cuda()))
print("TF is GPU available: " +str(tf.test.is_gpu_available()))
print("TF version: " +str(tf.__version__))

