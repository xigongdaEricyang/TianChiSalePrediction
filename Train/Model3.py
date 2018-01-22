import tensorflow as tf
import mnist_inference
import os
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.model_selection import train_test_split

BATCH_SIZE = 200 
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.002
TRAINING_STEPS = 30001
# nepochs = 100


MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "model/{}".format(pd.Timestamp.now())[:16]
MODEL_NAME = "tianchi_model"
TENSORBOARD_LOG = 'tensor_board'

def train(trainX, trainY, testX, testY):
    dataSize = len(trainY)
    
    with tf.device('/gpu:0'):
        # 定义输入输出placeholder。
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        y = mnist_inference.inference(x, regularizer)
        global_step = tf.Variable(0, trainable=False)
    
        # 定义损失函数、学习率、滑动平均操作以及训练过程。
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        #cross_entropy_mean = tf.reduce_mean(cross_entropy)
        beginLoss = tf.reduce_mean(tf.abs(tf.subtract(y, y_)))
        #loss = beginLoss + tf.add_n(tf.get_collection('losses'))
        loss = beginLoss
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            dataSize / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()
        
        for i in range(TRAINING_STEPS):
            for start in range(0, len(trainX)-BATCH_SIZE, BATCH_SIZE):
                end = start + BATCH_SIZE
                if end <=len(trainX):
                    x_batch, y_batch = trainX[start: end], trainY[start: end]
                else: 
                    x_batch, y_batch = trainX[start: ], trainY[start: ]
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: x_batch, y_: y_batch})
                if start % len(trainX) == 0:
                    print("After", step," training step(s), loss on training batch is ", loss_value)
            if i % 5 == 0:
                test_loss = sess.run(loss, feed_dict={x:test_x, y_: test_y})
                print("rang", i, "current test loss value on test set is ", test_loss)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

df = pd.read_csv('../CleanData/Max_trainDataAfterClean.csv')
# random shift the df
df = df.sample(frac=1).reset_index(drop=True)

normalizeColumns = ['compartment','TR','displacement','price_level','power','level_id',
                    'cylinder_number','engine_torque','car_length','car_height','car_width','total_quality','equipment_quality',
                    'rated_passenger','wheelbase','front_track','rear_track']
leftDf = df.drop(normalizeColumns, axis =1 ).drop(['sale_quantity'], axis = 1)

normalizeDf = df[normalizeColumns]
normalizeDf = (normalizeDf-normalizeDf.min())/(normalizeDf.max()-normalizeDf.min())
inputDf = pd.concat([leftDf, normalizeDf], axis = 1)
inputX = inputDf.values
resultArray = df['sale_quantity'].values
inputY = resultArray.reshape((len(resultArray),1))

train_x, test_x, train_y, test_y = train_test_split(inputX, inputY, test_size=0.2, random_state=42)

train(train_x, train_y, test_x, test_y)