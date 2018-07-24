if __name__ == '__main__':
    import shutil
    import numpy as np
    import tensorflow as tf
    import overfitting as ov
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.model_selection import train_test_split
    # 构造Z_data
    Sh = 20
    x = np.linspace(-10, 10, Sh, dtype=np.float32) + np.zeros((Sh, 1), dtype=np.float32)
    y = np.linspace(-10, 10, Sh, dtype=np.float32).reshape(Sh, 1) + np.zeros((1, Sh), dtype=np.float32)
    noise = np.random.normal(0, 0.5, size=(Sh, Sh)).astype(np.float32)
    # 拟合马鞍面z_data
    z_data = (x * y + noise).reshape(Sh*Sh, 1)
    # 构造输入数据inp_data
    inp_data = np.array(np.zeros((Sh*Sh, 2)), dtype=np.float32)
    inp_data[:, 0] = x.reshape((Sh*Sh, ))
    inp_data[:, 1] = y.reshape((Sh*Sh, ))
    # 划分训练集与测试集
    XY_train, XY_test, Z_train, Z_test = train_test_split(inp_data, z_data, test_size=0.3)
    # 构造Batch
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('inp_data'):
        xys = tf.placeholder(tf.float32, [None, 2])
    with tf.name_scope('out_data'):
        zs = tf.placeholder(tf.float32, [None, 1])
    # 构造神经网络
    L1 = ov.layer(xys, 2, 30, lay_name='Lay_one', keep_prob=keep_prob, Act_function=tf.nn.relu)
    L2 = ov.layer(L1, 30, 50, lay_name='Lay_two', keep_prob=keep_prob, Act_function=tf.nn.tanh)
    # L3 = ov.layer(L2, 50, 100, lay_name='Lay_three', keep_prob=keep_prob, Act_function=tf.nn.relu)
    z_pre = ov.layer(L2, 50, 1, lay_name='Lay_out', keep_prob=keep_prob, Act_function=None)
    # 构造损失函数与训练
    loss = ov.loss_step(zs, z_pre)
    train = ov.train_step(loss=loss)
    sess = ov.sess_step(2)
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    # 若存在log文件则删除
    try:
        shutil.rmtree('log')
    except FileNotFoundError:
        print("Can't find log")
    else:
        pass
    # 重写log文件
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    test_writer = tf.summary.FileWriter('log/test', sess.graph)
    i = -1
    l_loss = 100
    l_train = 100
    while (l_loss > 20 or l_train > 65) and i < 100000:
        i += 1
        sess.run(train, feed_dict={xys: XY_train, zs: Z_train, keep_prob: 0.6})
        l_train = sess.run(loss, feed_dict={xys: XY_train, zs: Z_train, keep_prob: 1})
        l_test = sess.run(loss, feed_dict={xys: XY_test, zs: Z_test, keep_prob: 1})
        l_loss = np.abs(l_train - l_test)
        if i % 50 == 0:
            train_result = sess.run(merged, feed_dict={xys: XY_train, zs: Z_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xys: XY_test, zs: Z_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
            # print(i, l_test)
    # 显示数据拟合曲面
    fig = plt.figure(0)
    axis = plt.subplot(111, projection='3d')
    axis.scatter(x, y, z_data.reshape(Sh, Sh))
    z = sess.run(z_pre, feed_dict={xys: inp_data, keep_prob: 1}).reshape(Sh, Sh)
    axis.plot_surface(x, y, z, cmap='rainbow')
    plt.savefig(r'./result/res.jpg', dpi=400)
    # 数据保存
    np.savetxt(r'./result/res.csv', z, delimiter=',')
