
# coding: utf-8

# In[1]:

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import shutil
import pandas as pd


# In[2]:

# 重新构建文件树
def organize_datasets(path_to_data, n=4000, ratio=0.2):
    files = os.listdir(path_to_data)
    files = [os.path.join(path_to_data, f) for f in files]
    shuffle(files)
    files = files[:n]

    n = int(len(files) * ratio)
    val, train = files[:n], files[n:]


    shutil.rmtree('./data/')
    print('/data/ removed')

    for c in ['dogs', 'cats']:
        os.makedirs('./data/train/{0}/'.format(c))
        os.makedirs('./data/validation/{0}/'.format(c))

    print('folders created !')

    for t in tqdm_notebook(train):
        if 'cat' in t:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'cats'))
        else:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'dogs'))

    for v in tqdm_notebook(val):
        if 'cat' in v:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'cats'))
        else:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'dogs'))

    print('Data copied!')

# 设置参数
ratio = 0.2
n = 25000
organize_datasets(path_to_data='D:\\1python notes\\cats vs dogs\\train', n=n, ratio=ratio)


# In[3]:

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback


# In[4]:

'''
创建两个 ImageDataGenerator 对象。
train_datagen 对应训练集，
val_datagen 对应测试集，两者都会对图像进行缩放，
train_datagen 还将做一些其他的修改
'''
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1 / 255.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )
val_datagen = ImageDataGenerator(rescale=1 / 255.)


# In[5]:

# 创建 train_generator and validation_generator
train_generator = train_datagen.flow_from_directory(
    './data/train/',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    './data/validation/',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')


# In[6]:

# 创建模型
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[7]:

# 训练步骤
epochs = 50
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[8]:

# 可视化模型
model.summary()


# In[9]:

'''
在训练模型前，我定义了两个将在训练时调用的回调函数 (callback function)：
一个用于在损失函数无法改进在测试数据的效果时，提前停止训练。
一个用于存储每个时期的损失和精确度指标：这可以用来绘制训练错误图表。
'''
## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

history = LossHistory()

## Callback for early stopping the training
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')


# In[10]:

'''
关于训练过程，还有几点要说的：
我们使用 fit_generator 方法，它是一个将生成器作为输入的变体（标准拟合方法）。
我们训练模型的时间超过 50 个 epoch。
'''
fitted_model = model.fit_generator(
    train_generator,
    steps_per_epoch=int(n * (1 - ratio)) // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=int(n * ratio) // batch_size,
    callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True), early_stopping, history],
    verbose=0)


# In[11]:

# 将模型保存
'''
epoch 2 [loss: 0.000, acc: 1.000, val_loss: 0.000, val_acc: 1.000] : 100% 625/625 [02:37<00:00, 4.22s/it]
'''
model.save('D:\\PycharmProjects\\project\\model4.h5')


# In[12]:

# 绘制训练和测试中的损失指标值
losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['loss'], 'g', label="train losses")
plt.plot(fitted_model.history['val_loss'], 'r', label="val losses")
plt.grid(True)
plt.title('Training loss vs. Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




# In[16]:

'''
当在两个连续的 epoch 中，测试损失值没有改善时，我们就中止训练过程
下面绘制训练集和测试集上的准确度
'''
losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['acc'], 'g', label="accuracy on train set")
plt.plot(fitted_model.history['val_acc'], 'r', label="accuracy on validation set")
plt.grid(True)
plt.title('Training Accuracy vs. Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




# In[17]:

'''
加载 VGG16 网络的权重，具体来说，
我要将网络权重加载到所有的卷积层。
这个网络部分将作为一个特征检测器来检测我们将要添加到全连接层的特征
'''

'''
start by loading the VGG16 weights (trained on ImageNet)
by specifying that we're not interested in the last three FC layers
'''
from keras import applications
# include_top: whether to include the 3 fully-connected layers at the top of the network.
model = applications.VGG16(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:




# In[19]:

'''
将图像传进网络来得到特征表示，
这些特征表示将会作为神经网络分类器的输入
'''
generator = datagen.flow_from_directory('./data/train/',
                                        target_size=(150, 150),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

bottleneck_features_train = model.predict_generator(generator, int(n * (1 - ratio)) // batch_size)
np.save(open('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


# In[21]:

generator = datagen.flow_from_directory('./data/validation/',
                                        target_size=(150, 150),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

bottleneck_features_validation = model.predict_generator(generator, int(n * ratio) // batch_size, )
np.save('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_validation.npy', bottleneck_features_validation)


# In[22]:

'''
图像在传递到网络中时是有序传递的，
所以我们可以很容易地为每张图片关联上标签
'''
train_data = np.load('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_train.npy')
train_labels = np.array([0] * (int((1-ratio) * n) // 2) + [1] * (int((1 - ratio) * n) // 2))

validation_data = np.load('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_validation.npy')
validation_labels = np.array([0] * (int(ratio * n) // 2) + [1] * (int(ratio * n) // 2))


# In[ ]:




# In[23]:

'''
现在我们设计了一个小型的全连接神经网络，
附加上从 VGG16 中抽取到的特征，
我们将它作为 CNN 的分类部分
'''
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:




# In[25]:

fitted_model = model.fit(train_data, train_labels,
                         epochs=25,
                         batch_size=batch_size,
                         validation_data=(validation_data, validation_labels[:validation_data.shape[0]]),
                         verbose=0,
                         callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=False), history])


# In[ ]:




# In[ ]:




# In[26]:

'''
在 25 个 epoch 后，模型就达到了 90.9% 的准确度
'''
# validation loss
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['loss'], 'g', label="train losses")
plt.plot(fitted_model.history['val_loss'], 'r', label="val losses")
plt.grid(True)
plt.title('Training loss vs. Validation loss - VGG16')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[ ]:




# In[27]:

# accuracy on validation set
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['acc'], 'g', label="accuracy on train set")
plt.plot(fitted_model.history['val_acc'], 'r', label="accuracy on validation set")
plt.grid(True)
plt.title('Training Accuracy vs. Validation Accuracy - VGG16')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[ ]:



