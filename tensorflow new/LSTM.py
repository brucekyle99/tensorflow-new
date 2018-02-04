import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST', one_hot=True)


#输入图片是28x28
n_inputs=28 #一行28个数据
max_time=28 #一共28行
lstm_size=100 #隐藏层单元
n_classes=10 #10个分类
batch_size=50 #每个批次50个样本
n_batch=mnist_data.train.num_examples // batch_size #总共的批次

#输入
x=tf.placeholder(tf.float32, [None, 784])
#实际的标签
y=tf.placeholder(tf.float32, [None, 10])


#初始化
weights=tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))  #权重是100x10
#初始化偏置
# biases=tf.Variables(tf.zeros([10]))
biases=tf.Variable(tf.constant(0.1, shape=[n_classes])) #10个偏置

#定义RNN网络
def RNN(X, weights, biases):
    #input=[batch_size, max_time, n_inputs]
    inputs=tf.reshape(X, [-1, max_time, n_inputs]) #input改为batch_size*max_time*n_inputs
    # 定义LSTM cell
    lstm_cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[state, batch_size （每个批次大小）, cell.state_size （隐藏单元个数）]
    # final_state[0] is cell state
    # final_state[1] is hidden_state
    # outputs: the RNN output 'tensor'
        # if time_major==false (default), this will be a 'tensor ' shaped:
        #         [batch_size, max_time, cell.output_size]
        # if time_major==True, this will be a tensor shaped:
        #          [max_time, batch_size, cell.output_size]
    outputs, final_state=tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1], weights)+biases)
    return results

# RNN 返回结果
prediction=RNN(x, weights, biases)
# loss function
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# optimizer
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#存放结果道bool表
correct_prediction=tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #把correct prediction变成float32类型
# 初始化
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc=sess.run(accuracy, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels})
        print('epoch ' + str(epoch) + ': test accuracy ' + str(acc))

'''
epoch 5: test accuracy 0.9733
'''















































