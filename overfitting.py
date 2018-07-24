import tensorflow as tf


def layer(Inp, Inp_size, Neuron_num, lay_name, keep_prob=1, Act_function=None):
    with tf.name_scope(str(lay_name)):
        with tf.name_scope('Weight'):
            Weight = tf.Variable(tf.random_normal([Inp_size,Neuron_num],dtype=tf.float32,seed=1),name='Weight')
            tf.summary.histogram('Weight',Weight)
        with tf.name_scope('Basic'):
            Basic = tf.Variable(tf.random_normal([1,Neuron_num],dtype=tf.float32,seed=1),name='Basic')
            tf.summary.histogram('Basic',Basic)
        with tf.name_scope('Out_Put'):
            Outputs_temp = tf.nn.dropout(tf.matmul(Inp,Weight)+ Basic,keep_prob)
            if Act_function is None:
                Outputs = Outputs_temp
            else:
                Outputs = Act_function(Outputs_temp)
            tf.summary.histogram('Output',Outputs)

    return Outputs


def loss_step(Z_data,Z_pre):
    with tf.name_scope('Loss'):
        loss_pre = tf.reduce_mean(tf.reduce_sum(tf.square(Z_data-Z_pre),reduction_indices=[1]))
        tf.summary.scalar('Loss',loss_pre)
    return loss_pre


def train_step(loss):
    with tf.name_scope('Train'):
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    return train


def sess_step(mode = 1,proportion = 0.333):
    if mode == 1:
        #训练方式为指定使用一定比例的Gpu容量
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=proportion)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    elif mode == 2:
        #训练方式为按使用增长所占Gpu容量
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        #使用cpu训练模型
        sess = tf.Session()
    return sess