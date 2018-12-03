import tensorflow as tf
import tensorflow.contrib as tfc
from utils import *
import time
import os
import numpy as np


class SegDepthModel:
    def __init__(self, sess, height=128, weight=448):
        self.reg = 1e-4  
        self.height = height
        self.weight = weight
        self.sess = sess

    def build_SegDepthModel(self):
        self.left = tf.placeholder(tf.float32, shape=[None, self.height, self.weight, 3], name='left_img')
        self.right = tf.placeholder(tf.float32, shape=[None, self.height, self.weight, 3], name='right_img')
        self.seg = tf.placeholder(tf.float32, shape=[None, self.height, self.weight, 3], name='seg_img')
        self.depth = tf.placeholder(tf.float32, shape=[None, self.height, self.weight, 3], name='depth_img')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # tf.add_to_collection("left", left)
        # tf.add_to_collection("right", right)
        # tf.add_to_collection("seg", seg)
        # tf.add_to_collection("depth", depth)
        
        # self.image_size_tf = tf.shape(self.left)[1:3]

        print('conv4_left:')
        conv4_left = self.CNN_res(self.left, filters=[32, 64], name='cnnleft')
        print('')
    
        print('u_left:')
        u_left = self.Unet(conv4_left, 32)
        print('')

        print('conv4_right:')
        conv4_right = self.CNN_res(self.right, filters=[32, 64], reuse=True, name='cnnleft')
        print('')

        print('u_right:')
        u_right = self.Unet(conv4_right, 32, reuse=True)
        print('')

        cnn_u = self.CNN_res(u_right, filters=[128, 64], name='cnn_u')
        cnn_u = conv_block(tf.layers.conv2d, cnn_u, 32, 1, strides=1, dilation_rate=1, name='cnn_u1', reg=self.reg)
        print('seg output:')
        seg_res = conv_block(tf.layers.conv2d, cnn_u, 3, 1, strides=1, dilation_rate=1, name='seg_res', reg=self.reg)
        print(seg_res.shape)
        print('')
        
        print('merge:')
        merge = self.lrMerge(u_left, u_right)
        print('')

        depth_cnn = self.CNN_res(merge, filters=[128, 64], name='depth_spp')
        depth_cnn = conv_block(tf.layers.conv2d, depth_cnn, 32, 1, strides=1, dilation_rate=1, name='depth_cnn',
                               reg=self.reg)
        print('depth output:')
        depth_res = conv_block(tf.layers.conv2d, depth_cnn, 3, 1, strides=1, dilation_rate=1, name='depth_cnn1',
                               reg=self.reg)
        print(depth_res.shape)
        print('')
        
        tf.add_to_collection("cseg_res", seg_res)
        tf.add_to_collection("cdepth_stack", depth_res)
        
        # self.loss = 0.5 * self.smooth_l1_loss(depth_stack, self.depth) + self.dice_loss(seg_res, self.seg) + \
        # 0.5 * self.mse(depth_stack, self.depth)
        self.loss = 0.5 * self.huber(depth_res, self.depth) + \
                    0.5 * self.smooth_l1_loss(depth_res, self.depth) + \
                    0.5 * self.rmse(seg_res, self.seg) + 0.5 * self.mae(seg_res, self.seg)
            
        tf.add_to_collection("closs", self.loss)

    def smooth_l1_loss(self, pred, targets, sigma=1.0):
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        sigma_2 = sigma ** 2
        box_diff = pred - targets
        abs_in_box_diff = tf.abs(box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * \
                      (1. - smoothL1_sign)
        loss_box = tf.reduce_mean(in_loss_box)
        
        return loss_box

    def rmse(self, pred, targets):
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        loss = tf.sqrt(tf.losses.mean_squared_error(targets, pred))
        return loss

    def mae(self, pred, targets):
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        loss = tf.losses.absolute_difference(targets, pred)
        return loss

    def huber(self, pred, targets):
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        loss = tf.losses.huber_loss(targets, pred)
        return loss

    def dice_loss(self, pred, targets):
        smooth = 1.
        # Flatten
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        intersection = tf.reduce_sum(tf.multiply(targets, pred))
        loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(targets) + tf.reduce_sum(pred) + smooth)
        return loss

    def CNN_res(self, bottom, filters=(32, 64, 128, 256), reuse=False, name='CNN_res'):
        dep = len(filters)
        print('Cnn_res'+' bottom:')
        print(bottom.shape)
        with tf.variable_scope(name):
            with tf.variable_scope('conv1'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[0], 1, strides=1, dilation_rate=1,
                                    name='conv1_0', reuse=reuse, reg=self.reg)
                for i in range(1):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[0], 3, dilation_rate=1,
                                       name='conv1_%d' % (i+1), reuse=reuse, reg=self.reg)
            if dep < 2:
                return bottom
            with tf.variable_scope('conv2'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[1], 1, strides=1, dilation_rate=1,
                                    name='conv2_0', reuse=reuse, reg=self.reg)
                for i in range(1):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[1], 3, dilation_rate=1,
                                       name='conv2_%d' % (i+1), reuse=reuse, reg=self.reg)
            if dep < 3:
                return bottom
            with tf.variable_scope('conv3'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[2], 1, strides=1, dilation_rate=1,
                                    name='conv3_0', reuse=reuse, reg=self.reg)
                for i in range(1):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[2], 3, dilation_rate=1,
                                       name='conv3_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
            if dep < 4:
                return bottom
            with tf.variable_scope('conv4'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[3], 1, strides=1, dilation_rate=1,
                                    name='conv4_0', reuse=reuse, reg=self.reg)
                for i in range(1):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[3], 3, dilation_rate=1,
                                       name='conv4_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
        print('Cnn_res_output:')
        print(bottom.shape)
        return bottom

    def SPP(self, bottom, reuse=False, name='spp'):
        print('SPP'+' bottom:')
        print(bottom.shape)
        with tf.variable_scope(name):
            branches = []
            for i, p in enumerate([16, 8, 4]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 32, 1, dilation_rate=1,
                                           name='branch_%d' % (i+1), reuse=reuse,
                                           reg=self.reg))
            concat = tf.concat(branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                fusion = conv_block(tf.layers.conv2d, concat, 32, 1, dilation_rate=1, name='conv2', reuse=reuse,
                                    reg=self.reg)
        print('Spp_output:')
        print(fusion.shape)
        return fusion
    
    def Unet(self, bottom, filters=32, reuse=False):
        kernel_size = 3
        pool_size = 2
        strides = 1
        dilation_rate = 1
        print('Unet'+' bottom:')
        print(bottom.shape)
        with tf.variable_scope('Unet'):
            downconv = []
            for i in range(4):
                bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool00'+str(i))
                bottom = conv_block(tf.layers.conv2d, bottom, filters, 1, strides, dilation_rate,
                                    name='dconv0'+str(i), reuse=reuse)
                bottom = res_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, dilation_rate,
                                   name='dconv1' + str(i), reuse=reuse)
                bottom = res_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, dilation_rate,
                                   name='dconv2'+str(i), reuse=reuse)
                downconv.append(bottom)
                filters = filters*2
                print(bottom.shape)
            
            bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool01'+'center')
            center = conv_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, dilation_rate,
                                name='cconv0'+'center', reuse=reuse)
            center = res_block(tf.layers.conv2d, center, filters, kernel_size, strides, dilation_rate,
                               name='dconv1center', reuse=reuse)
            center = res_block(tf.layers.conv2d, center, filters, kernel_size, strides, dilation_rate,
                               name='dconv2center', reuse=reuse)
            for i in range(4):
                filters = filters//2
                center = conv_block(tf.layers.conv2d_transpose, center, filters, kernel_size, 2, name='dconv'+str(i),
                                    reuse=reuse)
                print(center.shape)
                center = tf.concat([center, downconv[3-i]], axis=-1, name='uconcat'+str(i))
                center = conv_block(tf.layers.conv2d, center, filters, 1, strides, dilation_rate,
                                    name='uconv0'+str(i), reuse=reuse)
                center = res_block(tf.layers.conv2d, center, filters, kernel_size, strides, dilation_rate,
                                   name='uconv1'+str(i), reuse=reuse)
                center = res_block(tf.layers.conv2d, center, filters, kernel_size, strides, dilation_rate,
                                   name='uconv2' + str(i), reuse=reuse)
            center = conv_block(tf.layers.conv2d_transpose, center, filters, kernel_size, 2, name='oconv1'+'center',
                                reuse=reuse)
        print('Unet_output:')
        print(center.shape)
        return center

    def lrMerge(self, left_fmap, right_fmap):
        with tf.variable_scope('Merge'):
            print('Merge'+' bottom:')
            print(left_fmap.shape)
            lr_res = tf.add(left_fmap, right_fmap, name='lr_res')
            lr_concat = tf.concat([lr_res, left_fmap, right_fmap], axis=-1, name='lr_concat')
            print('Merge_output:')
            print(lr_concat.shape)
        return lr_concat

    def train(self, train_data_path, save_path, batch_size=10, val_ratio=0.1, learning_rate=0.001, epochs=5,
              fine_tune=False, stop_patience=3):

        ckpt_path = os.path.join(save_path, 'model.ckpt')
        train_saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2, allow_empty=True)

        leftimg_path_array = img_path_array(train_data_path[0], val_ratio, '0')
        rightimg_path_array = img_path_array(train_data_path[1], val_ratio, '1')
        segimg_path_array = img_path_array(train_data_path[2], val_ratio)
        depthimg_path_array = img_path_array(train_data_path[3], val_ratio)

        if val_ratio != 0:
            val_left = load_batch_img(leftimg_path_array[1], img_shape=(self.height, self.weight))
            val_right = load_batch_img(rightimg_path_array[1], img_shape=(self.height, self.weight))
            val_seg = load_batch_img(segimg_path_array[1], img_shape=(self.height, self.weight),
                                     prepro=False, label=1)
            val_depth = load_batch_img(depthimg_path_array[1], img_shape=(self.height, self.weight),
                                       prepro=False, label=2)
            
            leftimg_path_array = leftimg_path_array[0]
            rightimg_path_array = rightimg_path_array[0]
            segimg_path_array = segimg_path_array[0]
            depthimg_path_array = depthimg_path_array[0]

        num_of_train_samples = len(leftimg_path_array)
        random_index = np.arange(num_of_train_samples)
        np.random.shuffle(random_index)

        learning_rate_dacay = 0.5
        ada_learning_rate = tf.Variable(learning_rate)

        if fine_tune:
            AdagradOptimizer = tf.train.AdagradOptimizer(learning_rate=ada_learning_rate)
            train_optimizer = AdagradOptimizer.minimize(self.loss)
            self.sess.run(tf.initializers.global_variables())
            self.reload_SegDepthModel(save_path)
        else:
            RMSPropOptimizer = tf.train.RMSPropOptimizer(learning_rate=ada_learning_rate)
            train_optimizer = RMSPropOptimizer.minimize(self.loss)
            self.sess.run(tf.initializers.global_variables())
        aug_rate = 1*num_of_train_samples
        # augment random degree
        ran_aug = np.around(np.random.rand(aug_rate), decimals=2)
        # random select augment
        ran_sel = np.around(np.random.rand(aug_rate), decimals=2)

        start_time = time.time()
        localtime = time.strftime("%Y-%m-%d-%H-%M")

        log = open(save_path+'/tf_training_log_' + localtime + '.txt', "a+")
        epoch = 0
        last_loss = 500
        last_metric = 500
        p_count = 0
        while epoch <= epochs:
            ran = epoch * num_of_train_samples % aug_rate
            print('Epoch %d ...' % epoch)
            print('')
            total_loss = 0
            for step in range(0, num_of_train_samples, batch_size):

                batch_left = load_batch_img(leftimg_path_array, img_shape=(self.height, self.weight),
                                            random_index=random_index[step:step+batch_size],
                                            ran_aug=ran_aug[ran], ran_sel=ran_sel[ran])
                batch_right = load_batch_img(rightimg_path_array, img_shape=(self.height, self.weight),
                                             random_index=random_index[step:step+batch_size],
                                             ran_aug=ran_aug[ran], ran_sel=ran_sel[ran])
                batch_seg = load_batch_img(segimg_path_array, img_shape=(self.height, self.weight),
                                           random_index=random_index[step:step+batch_size],
                                           ran_aug=ran_aug[ran], ran_sel=ran_sel[ran], label=1, prepro=False)
                batch_depth = load_batch_img(depthimg_path_array, img_shape=(self.height, self.weight),
                                             random_index=random_index[step:step+batch_size],
                                             ran_aug=ran_aug[ran], ran_sel=ran_sel[ran], label=2, prepro=False)

                _, loss = self.sess.run([train_optimizer, self.loss],
                                        feed_dict={self.left: batch_left, self.right: batch_right,
                                                   self.seg: batch_seg, self.depth: batch_depth,
                                                   self.is_training: True})
                total_loss += loss
                slog = 'Step %d training loss = %.3f , time = %.2f' % (step, loss, time.time() - start_time)
                log.write(slog + '\n')
                # print(slog)
                # print('')
                ran += 1

            ave_loss = total_loss / num_of_train_samples * batch_size
            if val_ratio:
                metric = self.sess.run(self.loss,
                                        feed_dict={self.left: val_left, self.right: val_right,
                                                   self.seg: val_seg, self.depth: val_depth,
                                                   self.is_training: True})

                elog = 'Epoch %d metrics = %.3f ,ave_training loss = %.3f \n' % (epoch, metric, ave_loss)
                print(elog)
                log.write('\n' + elog + '\n\n')

            if metric < last_metric:
                last_metric = metric
                if epochs - epoch < 2:
                    epochs += 3
            if ave_loss < last_loss:
                last_loss = ave_loss
                # save model
                train_saver.save(self.sess, ckpt_path)
            else:
                p_count += 1
                if p_count >= stop_patience:
                    ada_learning_rate = tf.multiply(ada_learning_rate, learning_rate_dacay)
                    print('Adjust learning rate....%3f' % self.sess.run(ada_learning_rate))
                    p_count = 0

            epoch += 1
        log.close()

    def reload_SegDepthModel(self, load_path):
        meta_path = os.path.join(load_path, 'model.ckpt.meta')
        ckpt_path = os.path.join(load_path, 'model.ckpt')
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.sess, ckpt_path)
        graph = tf.get_default_graph()
        self.left = graph.get_tensor_by_name('left_img:0')
        self.right = graph.get_tensor_by_name('right_img:0')
        self.seg = graph.get_tensor_by_name('seg_img:0')
        self.depth = graph.get_tensor_by_name('depth_img:0')
        self.seg_o = graph.get_collection('cseg_res')[0]
        self.depth_o = graph.get_collection('cdepth_stack')[0]
        self.is_training = graph.get_tensor_by_name('is_training:0')
        self.loss = graph.get_collection('closs')[0]

    def predict_SegDepthModel(self, left_img, right_img, load_path):
        left_img = np.array([img_preprocess(left_img)])
        right_img = np.array([img_preprocess(right_img)])

        self.reload_SegDepthModel(load_path)
        seg, depth = self.sess.run([self.seg_o, self.depth_o],
                                   feed_dict={self.left: left_img, self.right: right_img,
                                              self.is_training: False})
        return seg, depth


if __name__ == '__main__':
    with tf.Session() as sess:
        model = SegDepthModel(sess)
