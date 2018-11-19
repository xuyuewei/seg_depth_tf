import tensorflow as tf
import os
import numpy as np

def load_jpeg(image_path,resize = (448,128)):
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_images(image, resize,align_corners=True,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    return image


def train_data_iterator(left_images_path,right_images_path,seg_path,depth_path,img_shape =(448,128),
                        batch_size=5,val_ratio = 0.1,shuffle=True,augment = True):
    #load data
    left_img_array = [os.path.join(left_images_path,x) for x in os.listdir(left_images_path)  if os.path.splitext(x)[0][-1]=='0']
    right_img_array = [os.path.join(right_images_path,x) for x in os.listdir(right_images_path)  if os.path.splitext(x)[0][-1]=='1']
    seg_array = [os.path.join(seg_path,x) for x in os.listdir(seg_path)]
    depth_array = [os.path.join(depth_path,x) for x in os.listdir(depth_path)]
    
    if val_ratio:
        num_of_samples = len(left_img_array)
        
        val_left_img_array = left_img_array[:np.int(val_ratio*num_of_samples)]
        val_right_img_array = right_img_array[:np.int(val_ratio*num_of_samples)]
        val_seg_array = seg_array[:np.int(val_ratio*num_of_samples)]
        val_depth_array = depth_array[:np.int(val_ratio*num_of_samples)]
        
        left_img_array = left_img_array[np.int(val_ratio*num_of_samples):]
        right_img_array = right_img_array[np.int(val_ratio*num_of_samples):]
        seg_array = seg_array[np.int(val_ratio*num_of_samples):]
        depth_array = depth_array[np.int(val_ratio*num_of_samples):]
        
        num_of_val_samples = len(val_left_img_array)
        val_data = tf.data.Dataset.from_tensor_slices((val_left_img_array,val_right_img_array,val_seg_array,val_depth_array))
        val_data = val_data.map(lambda x,y,z,w:(load_jpeg(x,img_shape),load_jpeg(y,img_shape),load_jpeg(z,img_shape),load_jpeg(w,img_shape)))
        
        val_data = val_data.batch(num_of_val_samples)
        val_data_iterator = val_data.make_initializable_iterator()

    train_data = tf.data.Dataset.from_tensor_slices((left_img_array,right_img_array,seg_array,depth_array))
    train_data = train_data.map(lambda x,y,z,w:(load_jpeg(x,img_shape),load_jpeg(y,img_shape),load_jpeg(z,img_shape),load_jpeg(w,img_shape)))
    
    if augment:
        train_data = train_data.map(augment)
    
    if shuffle:
        train_data = train_data.shuffle(num_of_samples)
    #repeat data for all epochs indefinitely
    train_data = train_data.repeat()
    train_data = train_data.prefetch(buffer_size=batch_size * 10).batch(batch_size)
    
    train_data_iterator = train_data.make_initializable_iterator()
    
    if val_ratio:
        return train_data_iterator,val_data_iterator
    return train_data_iterator

def predict_data_iterator(left_images_path,right_images_path,img_shape =(448,128),batch_size=1):
    #load data
    left_img_array = [os.path.join(left_images_path,x) for x in os.listdir(left_images_path)  if os.path.splitext(x)[0][-1]=='0']
    right_img_array = [os.path.join(right_images_path,x) for x in os.listdir(right_images_path)  if os.path.splitext(x)[0][-1]=='1']
    
    predict_data = tf.data.Dataset.from_tensor_slices((left_img_array,right_img_array))
    predict_data = predict_data.map(lambda x,y:(load_jpeg(x,img_shape),load_jpeg(y,img_shape)))
    predict_data = predict_data.batch(batch_size)
    
    predict_data_iterator = predict_data.make_initializable_iterator()
    return predict_data_iterator

def augment(left_input_img,right_input_img,seg_img,depth_img,
            hue_delta = 0.2,  # Adjust the hue of an RGB image by random factor
            brightness = 0.3,
            lsaturation = 0.1,
            usaturation = 0.3,
            angle = 60,
            projective_transform_angle = 60,
            img_size = [448,128],
            width_shift_range=0.3,  # Randomly translate the image horizontally
            height_shift_range=0.2):  # Randomly translate the image vertically
    
    left_input_img = tf.image.per_image_standardization(left_input_img)
    right_input_img = tf.image.per_image_standardization(right_input_img)
    seg_img = tf.image.per_image_standardization(seg_img)
    
    left_input_img,right_input_img,seg_img,depth_img = tf.image.random_hue(left_input_img,right_input_img,seg_img,depth_img, hue_delta)
    left_input_img,right_input_img,seg_img,depth_img = tf.image.random_brightness(left_input_img,right_input_img,seg_img,depth_img, brightness)
    left_input_img,right_input_img,seg_img,depth_img = tf.image.random_saturation(left_input_img,right_input_img,seg_img,depth_img, lsaturation,usaturation)       
    
    left_input_img,right_input_img,seg_img,depth_img = projective_random_transform(left_input_img,right_input_img,seg_img,depth_img,projective_transform_angle,img_size)
    
    left_input_img,right_input_img,seg_img,depth_img = shift_img(left_input_img,right_input_img,seg_img,depth_img, width_shift_range, height_shift_range,img_size)
    
    return left_input_img,right_input_img,seg_img,depth_img
    
    
def shift_img(left_input_img,right_input_img,seg_img,depth_img, width_shift_range, height_shift_range,img_size):
    """This fn will perform the horizontal or vertical shift"""
    img_shape = img_size
    if width_shift_range:
        width_shift_range = tf.random_uniform([], -width_shift_range * img_shape[1],
                                              width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],-height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
      # Translate all
    left_input_img = tf.contrib.image.translate(left_input_img,[width_shift_range, height_shift_range])
    right_input_img = tf.contrib.image.translate(right_input_img,[width_shift_range, height_shift_range])
    seg_img = tf.contrib.image.translate(seg_img,[width_shift_range, height_shift_range])
    depth_img = tf.contrib.image.translate(depth_img,[width_shift_range, height_shift_range])
        
    return left_input_img,right_input_img,seg_img,depth_img

def rot_randomangle(left_input_img,right_input_img,seg_img,depth_img, angle = 45):
    if angle:
        random_angle = tf.random_uniform([], 0.2, 1.0)*3.14*angle/180
        left_input_img = tf.contrib.image.rotate(left_input_img,random_angle)
        right_input_img = tf.contrib.image.rotate(right_input_img,random_angle)
        seg_img = tf.contrib.image.rotate(seg_img,random_angle)
        depth_img = tf.contrib.image.rotate(depth_img,random_angle)
        
    return left_input_img,right_input_img,seg_img,depth_img

def projective_random_transform(left_input_img,right_input_img,seg_img,depth_img,angle = 45,img_size = (448,128)):
    if angle:
        random_angle = tf.random_uniform([], 0.5, 1.0)*angle
        transform = tf.contrib.image.angles_to_projective_transforms(random_angle,img_size[0],img_size[1])
        left_input_img = tf.contrib.image.transform(left_input_img,transform)
        right_input_img = tf.contrib.image.transform(right_input_img,transform)
        seg_img = tf.contrib.image.transform(seg_img,transform)
        depth_img = tf.contrib.image.transform(depth_img,transform)
    return left_input_img,right_input_img,seg_img,depth_img

def flipran_img(left_input_img,right_input_img,seg_img,depth_img):
    
    flip_prob = tf.random_uniform([], 0.0, 1.0)
    
    left_input_img= tf.cond(tf.less(flip_prob, 0.3),
                        lambda: (tf.image.flip_up_down(tf.image.flip_left_right(left_input_img)),
                                 lambda: (tf.cond(tf.less(flip_prob,0.6),
                                                  lambda:(tf.image.flip_up_down(left_input_img)),
                                                  lambda:(tf.image.flip_left_right(left_input_img))))))
    right_input_img= tf.cond(tf.less(flip_prob, 0.3),
                        lambda: (tf.image.flip_up_down(tf.image.flip_left_right(right_input_img)),
                                 lambda: (tf.cond(tf.less(flip_prob,0.6),
                                                  lambda:(tf.image.flip_up_down(right_input_img)),
                                                  lambda:(tf.image.flip_left_right(right_input_img))))))
    seg_img= tf.cond(tf.less(flip_prob, 0.3),
                        lambda: (tf.image.flip_up_down(tf.image.flip_left_right(seg_img)),
                                 lambda: (tf.cond(tf.less(flip_prob,0.6),
                                                  lambda:(tf.image.flip_up_down(seg_img)),
                                                  lambda:(tf.image.flip_left_right(seg_img))))))
    depth_img= tf.cond(tf.less(flip_prob, 0.3),
                        lambda: (tf.image.flip_up_down(tf.image.flip_left_right(depth_img)),
                                 lambda: (tf.cond(tf.less(flip_prob,0.6),
                                                  lambda:(tf.image.flip_up_down(depth_img)),
                                                  lambda:(tf.image.flip_left_right(depth_img))))))
    
    return left_input_img,right_input_img,seg_img,depth_img



def conv_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
               apply_bn=True, apply_relu=True):
    with tf.variable_scope(name):
        conv_params = {
            'padding': 'same',
            'kernel_initializer': tfc.layers.xavier_initializer(),
            'kernel_regularizer': tfc.layers.l2_regularizer(reg),
            'bias_regularizer': tfc.layers.l2_regularizer(reg),
            'name': 'conv',
            'reuse': reuse
        }
        #这里需要注意，转置卷积不能加上空洞卷积的值
        conv_params['dilation_rate'] = dilation_rate
        #if dilation_rate == -1:
        #    conv_params[]
        bottom = func(bottom, filters, kernel_size, strides, **conv_params)
        if apply_bn:
            bottom = tf.layers.batch_normalization(bottom,
                                                   training=tf.get_default_graph().get_tensor_by_name('is_training:0'),
                                                   reuse=reuse, name='bn')
        if apply_relu:
            bottom = tf.nn.relu(bottom, name='relu')
        return bottom


def res_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
              projection=False):
    with tf.variable_scope(name):
        short_cut = bottom
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, name='conv1', reuse=reuse,
                            reg=reg)
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, name='conv2', reuse=reuse, reg=reg,
                            apply_relu=False)
        if projection:
            short_cut = conv_block(func, short_cut, filters, kernel_size, strides, dilation_rate, name='conv3', 
                                   reuse=reuse, reg=reg)
        bottom = tf.add(bottom, short_cut, 'add')
        return bottom


def SPP_branch(func, bottom, pool_size, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None,
               reg=1e-4, apply_bn=True, apply_relu=True):
    with tf.variable_scope(name):
        size = tf.shape(bottom)[1:3]
        bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool')
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, 'conv', reuse, reg,
                            apply_bn, apply_relu)
        bottom = tf.image.resize_images(bottom, size)
    return bottom
