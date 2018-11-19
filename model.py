import tensorflow as tf
import tensorflow.contrib as tfc
from utils import *
import time
import os

class Seg_Depth_Model:

    def __init__(self, sess, height=128, weight=448, batch_size=2):
        self.reg = 1e-4  
        self.height = height
        self.weight = weight
        self.batch_size = batch_size
        self.sess = sess

    def build_model(self):
        self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3],name= 'left_img')
        self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3],name= 'right_img')
        self.seg = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3],name= 'seg_img')
        self.depth = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3],name= 'depth_img')
        
        self.image_size_tf = tf.shape(self.left)[1:3]

        conv4_left = self.CNN_res(self.left,filters = [32,64])
        fusion_left = self.SPP(conv4_left)
        u_left = self.Unet(fusion_left,64)

        conv4_right = self.CNN_res(self.right,filters = [32,64],reuse = True)
        fusion_right = self.SPP(conv4_right,reuse = True)
        u_right = self.Unet(fusion_right,64,reuse = True)
        
        seg_res = self.CNN_res(u_left,filters = [128,64,32])
        seg_res = conv_block(tf.layers.conv2d, seg_res, 3, 1, strides=1, name='seg_res', reg=self.reg)
        
        merge = self.lrMerge(u_left,u_right)
        merge = self.CNN_res(merge,filters = [128,256,128,64])
        depth_stack = self.stackedhourglass('2d',merge)
        depth_stack = conv_block(tf.layers.conv2d, depth_stack, 3, 1, strides=1, name='depth_stack', reg=self.reg)
        
        #print(self.disps.shape)
        self.loss = self.smooth_l1_loss(seg_res, self.depth) + self.dice_loss(seg_res,self.seg)
        
        tf.add_to_collection("cseg_res", seg_res)
        tf.add_to_collection("cdepth_stack", depth_stack)
        tf.add_to_collection("closs", self.loss)
        

        
        try:
          self.sess.run(tf.global_variables_initializer())
        except:
          self.sess.run(tf.initialize_all_variables())
        
    def smooth_l1_loss(self, pred, targets, sigma=1.0):
        pred = tf.reshape(pred,[-1])
        targets = tf.reshape(targets,[-1])
        sigma_2 = sigma ** 2
        box_diff = pred - targets
        in_box_diff = box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
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

    def CNN_res(self, bottom, filters = [32,64,128,256], reuse = False):
        dep = len(filters)
        with tf.variable_scope('CNN'):
            with tf.variable_scope('conv1'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[0], 3, strides=1, name='conv1_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[0], 3, name='conv1_%d' % (i+1), reuse=reuse, reg=self.reg)
            if dep < 2:
                return bottom
            with tf.variable_scope('conv2'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[1], 3, strides=1, name='conv2_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[1], 3, name='conv2_%d' % (i+1), reuse=reuse, reg=self.reg)
            if dep < 3:
                return bottom
            with tf.variable_scope('conv3'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[2], 3, strides=1, name='conv3_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[2], 3, dilation_rate=1, name='conv3_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
            if dep < 4:
                return bottom
            with tf.variable_scope('conv4'):
                bottom = conv_block(tf.layers.conv2d, bottom, filters[3], 3, strides=1, name='conv4_0', reuse=reuse, reg=self.reg)
                for i in range(2):
                    bottom = res_block(tf.layers.conv2d, bottom, filters[3], 3, dilation_rate=2, name='conv4_%d' % (i+1), reuse=reuse,
                                       reg=self.reg)
                    
        return bottom

    def SPP(self, bottom, reuse=False):
        with tf.variable_scope('SPP'):
            branches = []
            for i, p in enumerate([32, 16, 8, 4]):
                branches.append(SPP_branch(tf.layers.conv2d, bottom, p, 128, 3, name='branch_%d' % (i+1), reuse=reuse,
                                           reg=self.reg))
            concat = tf.concat(branches, axis=-1, name='concat')
            with tf.variable_scope('fusion'):
                bottom = conv_block(tf.layers.conv2d, concat, 64, 3, name='conv1', reuse=reuse, reg=self.reg)
                fusion = conv_block(tf.layers.conv2d, bottom, 32, 1, name='conv2', reuse=reuse, reg=self.reg)
        return fusion
    
    def Unet(self,bottom,filters = 32,reuse=False):
        kernel_size = 3
        pool_size = 2
        strides=1
        with tf.variable_scope('Unet'):
            downconv = []
            for i in range(3):
                bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool00'+str(i))
                bottom = conv_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, 'conv0'+str(i), reuse)
                bottom = res_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, 'rconv0'+str(i), reuse)
                downconv.append(bottom)
                filters = filters*2
            
            filters  = filters*2
            bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool01'+str(i))
            center = conv_block(tf.layers.conv2d, bottom, filters, kernel_size, strides, 'cconv0'+str(i), reuse)
            filters = filters//2
            for i in range(3):
                filters = filters//2
                center = conv_block(tf.layers.conv2d_transpose, center, filters, kernel_size, 2, 'dconv'+str(i), reuse)
                center = tf.concat([center,downconv[2-i]], axis=-1, name='uconcat')
                center = conv_block(tf.layers.conv2d, center, filters, kernel_size, strides, 'conv1'+str(i), reuse)
                center = res_block(tf.layers.conv2d, center, filters, kernel_size, strides, 'rconv1'+str(i), reuse)
                
            center = conv_block(tf.layers.conv2d_transpose, center, filters, kernel_size, 2, 'oconv1'+str(i), reuse)
        return center
    
    def lrMerge(self,left_fmap,right_fmap,reuse=False):
        with tf.variable_scope('Merge'):
            lr_res = tf.add(left_fmap, right_fmap, name='lr_res')
            lr_concat = tf.concat([lr_res,left_fmap,right_fmap], axis=-1, name='lr_concat')
        return lr_concat
        
    def stackedhourglass(strs, bottom, filters = [32,64,128], kernel_size = 3, reg=1e-4):
        with tf.variable_scope('stackedhourglass'):
            short_cuts = []
            regressions = []
            conv_func, deconv_func = (tf.layers.conv2d, tf.layers.conv2d_transpose) if strs == '2d' else (tf.layers.conv3d, tf.layers.conv3d_transpose)
            bottom = conv_block(conv_func, bottom, filters[0], kernel_size, strides=1,
                                name='stack_0_1', reg=reg)
            bottom = res_block(conv_func, bottom, filters[0], kernel_size, strides=1,
                               name='stack_0_2', reg=reg)
            short_cuts.append(bottom)
            #print(list(zip(filters_list, kernel_size_list, short_cut_list)))
            for i in range(3):
                bottom = tf.layers.average_pooling2d(bottom, 2, 2, 'same', name='0avg_pool')
                bottom = conv_block(conv_func, bottom, filters[1], kernel_size, strides=2, 
                                    name='stack_%d_1' % (i+1), reg=reg)
                if i == 0:
                    short_cuts.append(bottom)
                    short_cuts.append(bottom)
                else:
                    bottom = tf.add(bottom, short_cuts[2], name='stack_%d_s' % (i + 1))
                    
                bottom = tf.layers.average_pooling2d(bottom, 2, 2, 'same', name='1avg_pool')
                bottom = conv_block(conv_func, bottom, filters[2], kernel_size,
                                    name='stack_%d_2' % (i+1), reg=reg)
                
                #反卷积有问题,必须确定batch-size,height,weight才能正确运行
                bottom = conv_block(deconv_func, bottom, filters[1], kernel_size, strides=2, name='stack_%d_3' % (i+1), reg=reg)
                bottom = tf.add(bottom, short_cuts[1], name='rstack_%d' % (i+1))
                short_cuts[2] = bottom
                
                bottom = conv_block(deconv_func, bottom, filters[0], kernel_size, strides=2, name='stack_%d_4' % (i+1), reg=reg)
                bottom = tf.add(bottom, short_cuts[0], name='drstack_%d' % (i+1))
                
                regressions.append(bottom)
            
            output = tf.concat(regressions, axis=-1, name='stack_concat')
            output = conv_block(conv_func, output, filters[0], kernel_size, strides=1, name='constack', reg=reg)

        return output
   
    def train(self, train_data,save_path,val_data = None,learning_rate = 0.005, epochs = 5,steps_per_epoch = 25,retrain = False):
        ckpt_path = os.path.join(save_path,'checkpoint/model.ckpt')
        saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours = 2)
        
        if retrain:
            Adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_optimizer = Adam_optimizer.minimize(self.loss)
        else:
            RMSProp_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            train_optimizer = RMSProp_optimizer.minimize(self.loss)
        
        if val_data:
            next_val_data = val_data.get_next()
            val_left, val_right,val_seg, val_depth = self.sess.run(next_val_data)
            
        start_time = time.time()
        for epoch in range(1, epochs+1):  
            for step in range(1,steps_per_epoch):
                next_batch_data = train_data.get_next()
                
                batch_left, batch_right, batch_seg, batch_depth = self.sess.run(next_batch_data)
                       
                _,loss = self.sess.run([train_optimizer, self.loss],
                                       feed_dict={self.left: batch_left, self.right: batch_right,
                                                  self.seg: batch_seg, self.depth: batch_depth})
                print('Step %d training loss = %.3f , time = %.2f' %(step,loss, time.time() - start_time))
                #save model
                saver.save(sess, ckpt_path, global_step=step)
                if val_data:
                    if step % 5 == 0:
                        metrics = self.sess.run(self.loss,feed_dict={self.left: val_left, self.right: val_right,
                                                                     self.seg: val_seg, self.depth: val_depth})
                        print('Step %d metrics = %.3f ,training loss = %.3f' %(step, metrics, loss))
        
    def predict(self,img_data,load_path):
        meta_path = os.path.join(load_path,'checkpoint/model.ckpt.meta')
        ckpt_path = os.path.join(load_path,'checkpoint/model.ckpt')
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.sess, ckpt_path) 
        
        graph = tf.get_default_graph()
        seg_res = graph.get_collection('cseg_res')
        depth_stack = graph.get_collection('cdepth_stack')
        seg_predict = []
        depth_predict = []
        for b in img_data:
        #next_batch_data = img_data.get_next()
            batch_left, batch_right = self.sess.run(b)
            seg, depth = self.sess.run([seg_res,depth_stack],feed_dict={self.left: batch_left, self.right: batch_right})
            seg_predict.append(seg)
            depth_predict.append(depth)
            
        return seg_predict,depth_predict
    
    def finetune(self, train_data,save_path,val_data = None,learning_rate = 0.005, epochs = 5,steps_per_epoch = 25):
        ckpt_path = os.path.join(save_path,'checkpoint/model.ckpt')
        meta_path = os.path.join(save_path,'checkpoint/model.ckpt.meta')
        saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours = 2)
        
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(self.sess, ckpt_path) 
        
        graph = tf.get_default_graph()
        loss = graph.get_collection('closs')
        left_img = graph.get_tensor_by_name('left_img:0')
        right_img = graph.get_tensor_by_name('right_img:0')
        seg_img = graph.get_tensor_by_name('seg_img:0')
        depth_img = graph.get_tensor_by_name('depth_img:0')
        
        Adam_optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        train_Adam = Adam_optimizer.minimize(loss)
        
        if val_data:
            next_val_data = val_data.get_next()
            val_left, val_right,val_seg, val_depth = self.sess.run(next_val_data)
            
        start_time = time.time()
        for epoch in range(1, epochs+1):  
            for step in range(1,steps_per_epoch):
                next_batch_data = train_data.get_next()
                
                batch_left, batch_right, batch_seg, batch_depth = self.sess.run(next_batch_data)
                       
                _,loss = self.sess.run([train_Adam, loss],
                                       feed_dict={left_img: batch_left, right_img: batch_right,
                                                  seg_img: batch_seg, depth_img: batch_depth})
                print('Step %d training loss = %.3f , time = %.2f' %(step,loss, time.time() - start_time))
                #save model
                saver.save(sess, ckpt_path, global_step=step)
                if val_data:
                    if step % 5 == 0:
                        metrics = self.sess.run(loss,feed_dict={left_img: val_left, right_img: val_right,
                                                                seg_img: val_seg, depth_img: val_depth})
                        print('Step %d metrics = %.3f ,training loss = %.3f' %(step, metrics, loss))
        
if __name__ == '__main__':
    with tf.Session() as sess:
        model = Seg_Depth_Model(sess)
