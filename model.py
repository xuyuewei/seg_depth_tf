import tensorflow as tf
import tensorflow.contrib as tfc
from utils import *
import time
import os
import numpy as np

class SegDepthModel:

    def __init__(self, sess, height=128, weight=448, batch_size=10):
        self.reg = 1e-4  
        self.height = height
        self.weight = weight
        self.batch_size = batch_size
        self.sess = sess
        self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3], name='left_img')
        self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3], name='right_img')
        self.seg = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3], name='seg_img')
        self.depth = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3], name='depth_img')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def build_model(self):
        # left = tf.constant(left,dtype=tf.Tensor,name = 'left_tensor')
        # right = tf.constant(right,dtype=tf.Tensor,name = 'right_tensor')
        # seg = tf.constant(seg,dtype=tf.Tensor,name = 'seg_tensor')
        # depth = tf.constant(depth,dtype=tf.Tensor,name = 'depth_tensor')

        # tf.add_to_collection("left", left)
        # tf.add_to_collection("right", right)
        # tf.add_to_collection("seg", seg)
        # tf.add_to_collection("depth", depth)
        
        # self.image_size_tf = tf.shape(self.left)[1:3]
        
        
        print('conv4_left:')
        conv4_left = self.CNN_res(self.left, filters=[32, 64])
        print('')
        print('fusion_left:')
        fusion_left = self.SPP(conv4_left)
        print('')
    
        print('u_left:')
        u_left = self.Unet(fusion_left, 64)
        print('')
        
        print('conv4_right:')
        conv4_right = self.CNN_res(self.right, filters=[32, 64], reuse=True)
        print('')
        
        print('fusion_right:')
        fusion_right = self.SPP(conv4_right, reuse=True)
        print('')
        
        print('u_right:')
        u_right = self.Unet(fusion_right, 64, reuse=True)
        print('')
        
        print('seg_res:')
        seg_res = self.CNN_res(u_left , filters=[128, 64, 32])
        print('')
        
        print('seg output:')
        seg_res = conv_block(tf.layers.conv2d, seg_res, 3, 1, strides=1,dilation_rate = 1, name='seg_res', reg=self.reg)
        print(seg_res.shape)
        print('')
        
        print('merge:')
        merge = self.lrMerge(u_left, u_right)
        print('')
        
        print('depth_stack:')
        depth_stack = self.stackedhourglass(merge)
        print('')
        depth_stack = conv_block(tf.layers.conv2d, depth_stack, 32, 3, strides=1, dilation_rate=1, name='depth_cnn',
                                 reg=self.reg)
        print('depth output:')
        depth_stack = conv_block(tf.layers.conv2d, depth_stack, 3, 1, strides=1, dilation_rate=1, name='depth_stack',
                                 reg=self.reg)
        print(depth_stack.shape)
        print('')
        
        tf.add_to_collection("cseg_res", seg_res)
        tf.add_to_collection("cdepth_stack", depth_stack)
        
        self.loss = self.smooth_l1_loss(seg_res, self.depth) + self.dice_loss(seg_res, self.seg)
            
        tf.add_to_collection("closs", self.loss)

    def smooth_l1_loss(self, pred, targets, sigma=1.0):
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        sigma_2 = sigma ** 2
        box_diff = pred - targets
        in_box_diff = box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * \
                      (1. - smoothL1_sign)
        out_loss_box = in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
        
        return loss_box
    
    def dice_loss(self,pred, targets):
        smooth = 1.
        # Flatten
        pred = tf.reshape(pred, [-1])
        targets = tf.reshape(targets, [-1])
        intersection = tf.reduce_sum(targets * pred)
        loss = 1 - (2. * intersection + smooth) / (tf.reduce_sum(targets) + tf.reduce_sum(pred) + smooth)
        return loss

    def CNN_res(self, bottom, filters=(32, 64, 128, 256), reuse=False):
        dep = len(filters)
        print('Cnn_res'+' bottom:')
        print(bottom.shape)
        with tf.variable_scope(str(dep)+'CNN'):
            with tf.variable_scope(str(dep)+'conv1'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[0], 3, strides=1, dilation_rate=1,
                                    name=str(dep)+'conv1_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[0], 3, dilation_rate=1,
                                       name=str(dep)+'conv1_%d' % (i+1), reuse=reuse, reg=self.reg)
            if dep < 2:
                return bottom
            with tf.variable_scope(str(dep)+'conv2'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[1], 3, strides=1, dilation_rate=1,
                                    name=str(dep)+'conv2_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[1], 3, dilation_rate=1,
                                       name=str(dep)+'conv2_%d' % (i+1), reuse=reuse, reg=self.reg)
            if dep < 3:
                return bottom
            with tf.variable_scope(str(dep)+'conv3'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[2], 3, strides=1, dilation_rate=1,
                                    name=str(dep)+'conv3_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[2], 3, dilation_rate=1,
                                       name=str(dep)+'conv3_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
            if dep < 4:
                return bottom
            with tf.variable_scope(str(dep)+'conv4'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[3], 3, strides=1,dilation_rate=1,
                                    name=str(dep)+'conv4_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[3], 3, dilation_rate=1,
                                       name=str(dep)+'conv4_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
        print('Cnn_res_output:')
        print(bottom.shape)
        return bottom

    def SPP(self, bottom, reuse=False):
        print('SPP'+' bottom:')
        print(bottom.shape)
        with tf.variable_scope('SPP'):
            branches = []
            for i, p in enumerate([32, 16, 8, 4]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 128, 3,dilation_rate=1,
                                           name='branch_%d' % (i+1), reuse=reuse,
                                           reg=self.reg))
            concat = tf.concat(branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                bottom = conv_block(tf.layers.conv2d, concat, 64, 3, dilation_rate=1, name='conv1', reuse=reuse,
                                    reg=self.reg)
                fusion = conv_block(tf.layers.conv2d, bottom, 32, 1,dilation_rate=1, name='conv2', reuse=reuse,
                                    reg=self.reg)
        print('Spp_output:')
        print(fusion.shape)
        return fusion
    
    def Unet(self, bottom, filters=32, reuse=False):
        kernel_size = 3
        pool_size = 2
        strides = 1
        dilation_rate=1
        print('Unet'+' bottom:')
        print(bottom.shape)
        with tf.variable_scope('Unet'):
            downconv = []
            for i in range(4):
                bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool00'+str(i))
                bottom = conv_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, dilation_rate,
                                    name='conv0'+str(i), reuse=reuse)
                bottom = res_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, dilation_rate,
                                   name='rconv0'+str(i), reuse=reuse)
                downconv.append(bottom)
                filters = filters*2
                print(bottom.shape)
            
            bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool01'+'center')
            center = conv_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, dilation_rate,
                                name='cconv0'+'center', reuse=reuse)
            for i in range(4):
                filters = filters//2
                center = conv_block(tf.layers.conv2d_transpose, center, filters, kernel_size, 2, name='dconv'+str(i),
                                    reuse=reuse)
                print(center.shape)
                center = tf.concat([center, downconv[3-i]], axis=-1, name='uconcat'+str(i))
                center = conv_block(tf.layers.conv2d, center, filters, kernel_size, strides, dilation_rate,
                                    name='conv1'+str(i), reuse=reuse)
                center = res_block(tf.layers.conv2d, center, filters, kernel_size, strides, dilation_rate,
                                   name='rconv1'+str(i), reuse=reuse)
                
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
        
    def stackedhourglass(self, bottom, filters=(32, 64, 128), kernel_size=3, reg=1e-4):
        with tf.variable_scope('Stackedhourglass'):
            print('Stackedhourglass'+' bottom:')
            print(bottom.shape)
            short_cuts = []
            regressions = []
            bottom = conv_block(tf.layers.conv2d, bottom, filters[0], kernel_size, strides=1, dilation_rate=1,
                                name='stack_0_1', reg=reg)
            bottom = res_block(tf.layers.conv2d, bottom, filters[0], kernel_size, strides=1, dilation_rate=1,
                               name='stack_0_2', reg=reg)
            short_cuts.append(bottom)
            for i in range(3):
                bottom = tf.layers.average_pooling2d(bottom, 4, 4, 'same', name='0avg_pool')
                bottom = conv_block(tf.layers.conv2d, bottom, filters[1], kernel_size, strides=1, dilation_rate=1,
                                    name='stack_%d_1' % (i+1), reg=reg)
                if i == 0:
                    short_cuts.append(bottom)
                    short_cuts.append(bottom)
                else:
                    bottom = tf.add(bottom, short_cuts[2], name='stack_%d_s' % (i + 1))
                print(bottom.shape)
                    
                bottom = tf.layers.average_pooling2d(bottom, 4, 4, 'same', name='1avg_pool')
                bottom = conv_block(tf.layers.conv2d, bottom, filters[2], kernel_size, strides=1, dilation_rate=1,
                                    name='stack_%d_2' % (i+1), reg=reg)
                print(bottom.shape)

                bottom = conv_block(tf.layers.conv2d_transpose, bottom, filters[1], kernel_size, strides=4,
                                    name='stack_%d_3' % (i+1), reg=reg)
                bottom = tf.add(bottom, short_cuts[1], name='rstack_%d' % (i+1))
                short_cuts[2] = bottom
                print(bottom.shape)
                bottom = conv_block(tf.layers.conv2d_transpose, bottom, filters[0], kernel_size, strides=4,
                                    name='stack_%d_4' % (i+1), reg=reg)
                bottom = tf.add(bottom, short_cuts[0], name='drstack_%d' % (i+1))
                print(bottom.shape)
                regressions.append(bottom)
            
            output = tf.concat(regressions, axis=-1, name='stack_concat')
            output = conv_block(tf.layers.conv2d, output, 64, kernel_size, strides=1, dilation_rate=1, name='constack',
                                reg=reg)
        print('Stackedhourglass_output:')
        print(output.shape)
        return output
   
    def train(self, train_data_path, save_path, batch_size=10, val_ratio=0.1, learning_rate=0.005, epochs=5,
              retrain=False):
        ckpt_path = os.path.join(save_path, 'checkpoint/model.ckpt')
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2, allow_empty=True)
        # self.sess.run(train_data)
        
        leftimg_path_array, rightimg_path_array = img_path_array(train_data_path[0], val_ratio, '0'), \
                                                  img_path_array(train_data_path[1], al_ratio, '1')
        segimg_path_array, depthimg_path_array = img_path_array(train_data_path[2], val_ratio),\
                                                 img_path_array(train_data_path[3], val_ratio)
        num_of_train_samples = len(leftimg_path_array)
        if val_ratio:
            num_of_val_samples = len(leftimg_path_array[1])
            index = np.arange(num_of_val_samples)
            val_left = load_batch_img(leftimg_path_array[1], random_index=index)
            val_right = load_batch_img(rightimg_path_array[1], random_index=index)
            val_seg = load_batch_img(segimg_path_array[1], random_index=index)
            val_depth = load_batch_img(depthimg_path_array[1], random_index=index)
            
            num_of_train_samples = len(leftimg_path_array[0])
            leftimg_path_array = leftimg_path_array[0]
            rightimg_path_array = rightimg_path_array[0]
            segimg_path_array = segimg_path_array[0]
            depthimg_path_array = depthimg_path_array[0]
        # graph = tf.get_default_graph()
        # loss = graph.get_collection('closs')
        # random_batch_array_index
        num_of_train_samples = len(leftimg_path_array[0])
        random_index = np.arange(num_of_train_samples)
        np.random.shuffle(random_index)
        steps_per_epoch = num_of_train_samples//batch_size
        
        if retrain:
            Adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_optimizer = Adam_optimizer.minimize(self.loss)
        else:
            RMSProp_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_optimizer = RMSProp_optimizer.minimize(self.loss)
            
        init = tf.initializers.global_variables()
        self.sess.run(init)
        
        start_time = time.time()
        for epoch in range(1, epochs+1):  
            for step in range(steps_per_epoch):
                ran_aug = np.around(np.random.rand(), decimals=2)
                batch_left = load_batch_img(leftimg_path_array, random_index=random_index[step, batch_size],
                                            ran_aug=ran_aug)
                batch_right = load_batch_img(rightimg_path_array, random_index=random_index[step, batch_size],
                                             ran_aug=ran_aug)
                batch_seg = load_batch_img(segimg_path_array, random_index=random_index[step, batch_size],
                                           ran_aug=ran_aug)
                batch_depth = load_batch_img(depthimg_path_array, random_index=random_index[step, batch_size],
                                             ran_aug=ran_aug)

                _, loss = self.sess.run([train_optimizer, self.loss],
                                        feed_dict={self.left: batch_left, self.right: batch_right,
                                                   self.seg: batch_seg, self.depth: batch_depth,
                                                   self.is_training: True})
                print('Step %d training loss = %.3f , time = %.2f' % (step, loss, time.time() - start_time))
                # save model
                saver.save(self.sess, ckpt_path, global_step=step)
                if val_ratio:
                    if step % 1 == 0:
                        metrics = self.sess.run(self.loss, feed_dict={self.left: val_left, self.right: val_right,
                                                                      self.seg: val_seg, self.depth: val_depth,
                                                                      self.is_training: False})
                        print('Step %d metrics = %.3f ,training loss = %.3f \n' % (step+1, metrics, loss))
        
    def predict(self, img_data, load_path):
        meta_path = os.path.join(load_path, 'checkpoint/model.ckpt.meta')
        ckpt_path = os.path.join(load_path, 'checkpoint/model.ckpt')
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.sess, ckpt_path) 
        
        graph = tf.get_default_graph()
        
        seg_res = graph.get_collection('cseg_res')
        depth_stack = graph.get_collection('cdepth_stack')
        seg_predict = []
        depth_predict = []
        for b in img_data:
            batch_left, batch_right = self.sess.run(b)
            seg, depth = self.sess.run([seg_res, depth_stack],
                                       feed_dict={self.left: batch_left, self.right: batch_right,
                                                  self.is_training: False})
            seg_predict.append(seg)
            depth_predict.append(depth)
            
        return seg_predict, depth_predict
    
    def finetune(self, train_data_path, save_path, batch_size=10, val_ratio=0.1, learning_rate=0.005, epochs=5):
        ckpt_path = os.path.join(save_path, 'checkpoint/model.ckpt')
        meta_path = os.path.join(save_path, 'checkpoint/model.ckpt.meta')
        # saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)
        
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.sess, ckpt_path) 
        
        graph = tf.get_default_graph()
        left_img = graph.get_tensor_by_name('left_img:0')
        right_img = graph.get_tensor_by_name('right_img:0')
        seg_img = graph.get_tensor_by_name('seg_img:0')
        depth_img = graph.get_tensor_by_name('depth_img:0')
        training = graph.get_tensor_by_name('is_training:0')
        loss = graph.get_collection('closs')
        
        Adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_Adam = Adam_optimizer.minimize(loss)
        
        leftimg_path_array = img_path_array(train_data_path[0], val_ratio, '0')
        rightimg_path_array = img_path_array(train_data_path[1], val_ratio, '1')
        segimg_path_array = img_path_array(train_data_path[2], val_ratio)
        depthimg_path_array = img_path_array(train_data_path[3], val_ratio)
        num_of_train_samples = len(leftimg_path_array)
        
        if val_ratio:
            num_of_val_samples = len(leftimg_path_array[1])
            index = np.arange(num_of_val_samples)
            val_left = load_batch_img(leftimg_path_array[1], random_index=index)
            val_right = load_batch_img(rightimg_path_array[1], random_index=index)
            val_seg = load_batch_img(segimg_path_array[1], random_index=index)
            val_depth = load_batch_img(depthimg_path_array[1], random_index=index)
            
            num_of_train_samples = len(leftimg_path_array[0])
            leftimg_path_array = leftimg_path_array[0]
            rightimg_path_array = rightimg_path_array[0]
            segimg_path_array = segimg_path_array[0]
            depthimg_path_array = depthimg_path_array[0]
        # graph = tf.get_default_graph()
        # loss = graph.get_collection('closs')
        # random_batch_array_index
        random_index = np.arange(num_of_train_samples)
        np.random.shuffle(random_index)
        steps_per_epoch = num_of_train_samples//batch_size
        
        init = tf.initializers.global_variables()
        self.sess.run(init)
        
        start_time = time.time()
        for epoch in range(1, epochs+1):  
            for step in range(steps_per_epoch):
                ran_aug = np.around(np.random.rand(), decimals=2)
                batch_left = load_batch_img(leftimg_path_array, random_index=random_index[step, batch_size],
                                            ran_aug=ran_aug)
                batch_right = load_batch_img(rightimg_path_array, random_index=random_index[step, batch_size],
                                             ran_aug=ran_aug)
                batch_seg = load_batch_img(segimg_path_array, random_index=random_index[step, batch_size],
                                           ran_aug=ran_aug)
                batch_depth = load_batch_img(depthimg_path_array, random_index=random_index[step, batch_size],
                                             ran_aug=ran_aug)
                
                _, loss = self.sess.run([train_Adam, loss],
                                        feed_dict={left_img: batch_left, right_img: batch_right,
                                                   seg_img: batch_seg, depth_img: batch_depth, training: True})
                print('Step %d training loss = %.3f , time = %.2f' % (step, loss, time.time() - start_time))
                # save model
                saver.save(self.sess, ckpt_path, global_step=step)
                if val_ratio:
                    if step % 1 == 0:
                        metrics = self.sess.run(loss, feed_dict={left_img: val_left, right_img: val_right,
                                                                 seg_img: val_seg, depth_img: val_depth,
                                                                 training: False})
                        print('Step %d metrics = %.3f ,training loss = %.3f \n' % (step+1, metrics, loss))

        
if __name__ == '__main__':
    with tf.Session() as sess:
        model = Seg_Depth_Model(sess)