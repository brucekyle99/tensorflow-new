import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST', one_hot=True)

#each batch size
batch_size=50
#count whole batch number
number_of_batch=mnist_data.train.num_examples // batch_size


#define placeholder
x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32, [None, 10])
keep_prob=tf.placeholder(tf.float32)
lr=tf.Variable(0.001, dtype=tf.float32)


# 1 layer
w1=tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1=tf.Variable(tf.zeros([500])+0.1)
l1=tf.nn.tanh(tf.matmul(x, w1)+b1)
l1_drop=tf.nn.dropout(l1, keep_prob)

# #2 nd layer
w2=tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
l2=tf.nn.tanh(tf.matmul(l1_drop, w2)+b2)
l2_drop=tf.nn.dropout(l2, keep_prob)
#
# w3=tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
# b3=tf.Variable(tf.zeros([2000])+0.1)
# l3=tf.nn.tanh(tf.matmul(l2_drop, w3)+b3)
# l3_drop=tf.nn.dropout(l3, keep_prob)

w4=tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b4=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(l2_drop, w4)+b4)


# #1 layer only
# w=tf.Variable(tf.ones([784, 10]))
# b=tf.Variable(tf.ones([10]))
# prediction=tf.nn.softmax(tf.matmul(x, w)+b)


#define loss function
# loss=tf.reduce_mean(tf.square(prediction-y))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#definr train
# train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train_step=tf.train.AdamOptimizer(0.01).minimize(loss)
train_step=tf.train.AdamOptimizer(lr).minimize(loss)
# train_step=tf.train.AdadeltaOptimizer().minimize(loss)
# train_step=tf.train.RMSPropOptimizer(0.001).minimize(loss)
# train_step=tf.train.AdagradOptimizer(0.01).minimize(loss)


#init variables
init=tf.global_variables_initializer()

#save result as bool type list
#argmax() returns the indice of the max value
right_prediction=tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# calculate accuracy
accuracy=tf.reduce_mean(tf.cast(right_prediction, tf.float32))


# run
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(61):
        sess.run(tf.assign(lr, 0.001*(0.95**epoch)))
        for batch in range(number_of_batch):
            batch_x, batch_y=mnist_data.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        learning_rate=sess.run(lr)
        test_acc=sess.run(accuracy, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels, keep_prob: 1.0})
        # train_acc = sess.run(accuracy, feed_dict={x: mnist_data.train.images, y: mnist_data.train.labels, keep_prob: 1.0})
        # print('epoch '+str(epoch)+': test accuracy '+str(test_acc)+' train accuracy ' + str(train_acc))
        print('epoch ' + str(epoch) + ': test accuracy ' + str(test_acc) + ' learning_rate ' + str(learning_rate))
        # print('epoch ' + str(epoch) + ': train accuracy ' + str(train_acc))

'''
epoch 0: test accuracy 0.9507 learning_rate 0.001
epoch 1: test accuracy 0.9609 learning_rate 0.00095
epoch 2: test accuracy 0.9658 learning_rate 0.0009025
epoch 3: test accuracy 0.9706 learning_rate 0.000857375
epoch 4: test accuracy 0.973 learning_rate 0.000814506
epoch 5: test accuracy 0.9744 learning_rate 0.000773781
epoch 6: test accuracy 0.9711 learning_rate 0.000735092
epoch 7: test accuracy 0.9726 learning_rate 0.000698337
epoch 8: test accuracy 0.9743 learning_rate 0.00066342
epoch 9: test accuracy 0.9761 learning_rate 0.000630249
epoch 10: test accuracy 0.9772 learning_rate 0.000598737
epoch 11: test accuracy 0.9783 learning_rate 0.0005688
epoch 12: test accuracy 0.9801 learning_rate 0.00054036
epoch 13: test accuracy 0.9767 learning_rate 0.000513342
epoch 14: test accuracy 0.9794 learning_rate 0.000487675
epoch 15: test accuracy 0.9812 learning_rate 0.000463291
epoch 16: test accuracy 0.98 learning_rate 0.000440127
epoch 17: test accuracy 0.9813 learning_rate 0.00041812
epoch 18: test accuracy 0.9796 learning_rate 0.000397214
epoch 19: test accuracy 0.9808 learning_rate 0.000377354
epoch 20: test accuracy 0.9782 learning_rate 0.000358486
epoch 21: test accuracy 0.98 learning_rate 0.000340562
epoch 22: test accuracy 0.9811 learning_rate 0.000323534
epoch 23: test accuracy 0.9804 learning_rate 0.000307357
epoch 24: test accuracy 0.9804 learning_rate 0.000291989
epoch 25: test accuracy 0.9808 learning_rate 0.00027739
epoch 26: test accuracy 0.9811 learning_rate 0.00026352
epoch 27: test accuracy 0.9808 learning_rate 0.000250344
epoch 28: test accuracy 0.9818 learning_rate 0.000237827
epoch 29: test accuracy 0.9807 learning_rate 0.000225936
epoch 30: test accuracy 0.9822 learning_rate 0.000214639
epoch 31: test accuracy 0.9819 learning_rate 0.000203907
epoch 32: test accuracy 0.9824 learning_rate 0.000193711
epoch 33: test accuracy 0.9816 learning_rate 0.000184026
epoch 34: test accuracy 0.982 learning_rate 0.000174825
epoch 35: test accuracy 0.9821 learning_rate 0.000166083
epoch 36: test accuracy 0.9828 learning_rate 0.000157779
epoch 37: test accuracy 0.982 learning_rate 0.00014989
epoch 38: test accuracy 0.9813 learning_rate 0.000142396
epoch 39: test accuracy 0.9814 learning_rate 0.000135276
epoch 40: test accuracy 0.9831 learning_rate 0.000128512
epoch 41: test accuracy 0.9824 learning_rate 0.000122087
epoch 42: test accuracy 0.9829 learning_rate 0.000115982
epoch 43: test accuracy 0.9828 learning_rate 0.000110183
epoch 44: test accuracy 0.9824 learning_rate 0.000104674
epoch 45: test accuracy 0.9827 learning_rate 9.94403e-05
epoch 46: test accuracy 0.9824 learning_rate 9.44682e-05
epoch 47: test accuracy 0.9826 learning_rate 8.97448e-05
epoch 48: test accuracy 0.9829 learning_rate 8.52576e-05
epoch 49: test accuracy 0.983 learning_rate 8.09947e-05
epoch 50: test accuracy 0.9827 learning_rate 7.6945e-05
epoch 51: test accuracy 0.9826 learning_rate 7.30977e-05
epoch 52: test accuracy 0.9834 learning_rate 6.94428e-05
epoch 53: test accuracy 0.9827 learning_rate 6.59707e-05
epoch 54: test accuracy 0.9826 learning_rate 6.26722e-05
epoch 55: test accuracy 0.9833 learning_rate 5.95386e-05
epoch 56: test accuracy 0.9825 learning_rate 5.65616e-05
epoch 57: test accuracy 0.9836 learning_rate 5.37335e-05
epoch 58: test accuracy 0.9829 learning_rate 5.10469e-05
epoch 59: test accuracy 0.9837 learning_rate 4.84945e-05
epoch 60: test accuracy 0.9833 learning_rate 4.60698e-05
'''