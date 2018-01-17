import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
print(x_data)
print(x_data.shape)


noise=np.random.normal(0, 0.02, x_data.shape) #形状和x_data一样
y_data=np.square(x_data)+noise

x=tf.placeholder(tf.float32, [None, 1])
y=tf.placeholder(tf.float32, [None, 1])

#define middle layer
weights_l1=tf.Variable(tf.random_normal([1, 10]))
biases_l1=tf.Variable(tf.zeros([1, 10]))
l1_wx_plus_b=tf.matmul(x, weights_l1)+biases_l1
l1_output=tf.nn.tanh(l1_wx_plus_b)

#define output layer
weights_l2=tf.Variable(tf.random_normal([10, 1]))
biases_l2=tf.Variable(tf.zeros([1, 1]))
l2_wx_plus_b=tf.matmul(l1_output, weights_l2)+biases_l2
#output is l2_output
l2_output=tf.tanh(l2_wx_plus_b)

#loss function and train
loss=tf.reduce_mean(tf.square(y-l2_output))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #init variable
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    prediction_value=sess.run(l2_output, feed_dict={x: x_data})
    #plot result
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r--', lw=5)
    plt.show()







