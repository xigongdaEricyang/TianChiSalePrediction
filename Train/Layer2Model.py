import tensorflow as tf

INPUT_NODE = 245
OUTPUT_NODE = 1
LAYER1_NODE = 500
LAYER2_NODE = 100

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope('layer3'):
        weights = get_weight_variable([LAYER2_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    return layer3