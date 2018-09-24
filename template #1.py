import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import time
import logging


def prepare_data():

    # 生成训练数据
    yield train_x, train_y


def model():
    # define input
    model_input = tf.placeholder(tf.float32, shape=[None, 200, 32], name="model-input")

    # define model label
    y_label = tf.placeholder(tf.float32, shape=[None, 200, 1], name="y_label")

    # construct model
    hidden=fully_connected(model_input, 128)
    y_predict=fully_connected(hidden, 2)

    return model_input, y_predict, y_label


def train(file_train, file_test):
    # 预定义参数
    batch_size=1000
    epoch=10
    train_steps=10000
    print_step=10
    saver_step=100
    # steps = total_sample * epoch / batch_size

    # 设置GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 数据获取
    user_index_batch, score_batch = read_tf_batch(file_train, batch_size, is_shuffle=True)
    user_index_batch_test, score_batch_test = read_tf_batch(file_test, batch_size * 5,
                                                                                    is_shuffle=True)

    # 得到模型输出，定义loss，和优化器
    model_input, y_predict, y_label=model()
    loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_predict)
    optimizer=tf.train.AdamOptimizer().minimize(loss)

    # 如果要保存模型
    saver=tf.train.Saver()

    with tf.Session() as sess:
        # 初始化
        init=tf.global_variables_initializer()
        sess.run(init)

        # 多线程（需要的话）
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)

        # 开始 train,一共要训练 train_steps 轮
        for step in range(train_steps):
            # 计时
            time1=time.time()

            #加载训练数据
            user_matrix_data, score = sess.run([user_index_batch, score_batch]) #分批次输入训练数据
            train_dict = {model_input: user_matrix_data, y_label: score} # 训练数据的 placeholder
            sess.run(optimizer, train_dict)

            if step % print_step ==0:
                # 若干个训练批次后打印训练集上的 loss
                loss_a=sess.run(loss, feed_dict=train_dict)

                # 若干轮后，测试模型此时在测试集上的 loss
                # 得到批次:  test_inputs, test_labels = generate_batch(embeddings, test_data, batch_size, hidden)
                user_matrix_test, score_test = sess.run([user_index_batch_test, score_batch_test])
                test_dict = {model_input: user_matrix_test, y_label: score_test.reshape([len(score_test), 1])}
                # 测试 test loss
                loss_t = sess.run(loss, feed_dict=test_dict)
                # 显示此时的测试集loss
                print('step = ' + str(step) + ' loss_t = ' + str(loss_t))

            # 定时保存训练完的模型
            if step % saver_step==0:
                saverPath = saver.save(sess, '/home/dev/data/models/0827/FMLModel.ckpt',
                                       global_step=step)


        # 多线程操作
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    tf_train_file=''
    tf_test_file=''
    train(tf_train_file, tf_test_file)


























































