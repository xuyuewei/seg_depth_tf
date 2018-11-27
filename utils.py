import tensorflow as tf
import tensorflow.contrib as tfc
import os
import numpy as np
import cv2 as cv


def load_jpeg(image_path, resize=(128, 448)):

    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, resize,align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return image


def cvload_img(image_path, resize=(448, 128)):
    image = cv.imread(image_path)
    image = cv.resize(image, (resize[1], resize[0]), interpolation=cv.INTER_LINEAR)
    return image


def img_path_array(path, val_ratio=0.1, pattern=''):
    if pattern == '':
        path_array = [os.path.join(path, x) for x in os.listdir(path)]
        if val_ratio:
            train_path_array,val_path_array = train_val_split(path_array,val_ratio)
            return train_path_array, val_path_array
        else:
            return path_array
    else:
        path_array = [os.path.join(path, x) for x in os.listdir(path) if os.path.splitext(x)[0][-1]==pattern]
        if val_ratio:
            train_path_array, val_path_array = train_val_split(path_array, val_ratio)
            return train_path_array, val_path_array
        else:
            return path_array


def train_val_split(images_path_array, val_ratio=0.1):
    num_of_samples = len(images_path_array)
    val_rat = np.int(val_ratio*num_of_samples)
    val_img_array = np.array(images_path_array[:val_rat])
    train_img_array = np.array(images_path_array[:val_rat])
    return train_img_array,val_img_array


def load_batch_img(images_path_array, random_index, img_shape=(128, 448), ran_aug=None, ran_sel=0.1):
    
    if ran_aug is not None:
        batch_img = np.array(map(lambda x: (augment(load_jpeg(x, img_shape), ran_aug, ran_sel)), images_path_array[random_index]))
    else:
        batch_img = np.array(map(lambda x: load_jpeg(x, img_shape), images_path_array[random_index]))
    return batch_img


def augment(input_img, ran_degree, ran_aug,
            adjust_brightness = False,
            angle=180,
            img_size=(128, 448),
            width_shift_range=0.4,  # Randomly translate the image horizontally
            height_shift_range=0.3):  # Randomly translate the image vertically

    np_ran = ran_degree*2-1
    half_size = (img_size[1]//2, img_size[0]//2)
    quar_size = (img_size[1]//4, img_size[0]//4)
    eith_size = (img_size[1]//8, img_size[0]//8)
    
    # darkness improve
    sd = np.floor(255/np.log2(255))
    process_img = (sd*np.log2(input_img+0.0001)).astype(np.uint8)
    
    # gamma enhance contrast
    sg = np.floor(np.power(255,1.8)/255)
    process_img = (np.power(process_img,1.8)/sg).astype(np.uint8)
    
    if ran_aug < 0.5:
        # perspective_transform
        pts1 = np.float32([[quar_size[0], quar_size[1]], [3*quar_size[0], quar_size[1]],
                           [quar_size[0], 3*quar_size[1]], [3*quar_size[0], 3*quar_size[1]]])
        pts2 = np.float32([[quar_size[0]-np_ran*eith_size[0], quar_size[1]-np_ran*eith_size[1]],
                           [3*quar_size[0]+np_ran*eith_size[0], quar_size[1]-np_ran*eith_size[1]],
                           [quar_size[0]-np_ran*eith_size[0], 3*quar_size[1]+np_ran*eith_size[1]],
                           [3*quar_size[0]+np_ran*eith_size[0], 3*quar_size[1]+np_ran*eith_size[1]]])
        M_perspective = cv.getPerspectiveTransform(pts1, pts2)
        process_img = cv.warpPerspective(process_img, M_perspective, (img_size[1], img_size[0]),
                                       borderMode=cv.BORDER_REFLECT)
        # scale nad rotate by random
        M_rot = cv.getRotationMatrix2D((half_size[0]-eith_size[0]*np_ran, half_size[1]-eith_size[1]*np_ran),
                                       angle*ran_degree, 1)
        process_img = cv.warpAffine(process_img, M_rot, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
    else:
        #translate
        M_trans = np.float32([[1, 0, img_size[1]*width_shift_range*np_ran],
                                          [0, 1, img_size[0]*height_shift_range*np_ran]])
        process_img = cv.warpAffine(process_img, M_trans, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
        np_ran = np_ran / 2
        # affine_transform
        apts1 = np.float32([[quar_size[0], quar_size[1]], [3*quar_size[0], quar_size[1]],
                            [quar_size[0], 3*quar_size[1]]])
        apts2 = np.float32([[quar_size[0]+np_ran*eith_size[0], quar_size[1]+np_ran*eith_size[1]],
                            [3*quar_size[0]-np_ran*eith_size[0], quar_size[1]-np_ran*eith_size[1]],
                            [quar_size[0]-np_ran*eith_size[0], 3*quar_size[1]-np_ran*eith_size[1]]])
        M_affine = cv.getAffineTransform(apts1, apts2)
        process_img = cv.warpAffine(process_img, M_affine, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
        
    
    # adjust contrast
    if adjust_brightness:
        ran_degree = ran_degree*0.4+0.6
        # lower brightness
        process_img = (process_img*ran_degree).astype(np.uint8)
    return process_img


def conv_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=-1, name=None, reuse=None, reg=1e-4,
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
        # 这里需要注意，转置卷积不能加上空洞卷积的值
        if dilation_rate >-1:
            conv_params['dilation_rate'] = dilation_rate
        # if dilation_rate == -1:
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
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, name='conv2', reuse=reuse,
                            reg=reg, apply_relu=False)
        if projection:
            short_cut = conv_block(func, short_cut, filters, kernel_size, strides, dilation_rate, name='conv3', 
                                   reuse=reuse, reg=reg)
        bottom = tf.add(bottom, short_cut, 'add')
        return bottom


def SPP_branch(func, bottom, pool_size, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None,
               reg=1e-4, apply_bn=True, apply_relu=True):
    with tf.variable_scope(name):
        bottom = tf.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool')
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, 'conv', reuse, reg,
                            apply_bn, apply_relu)
        print('average_pooling_output:'+str(bottom.shape))
        bottom = conv_block(tf.layers.conv2d_transpose, bottom, filters, kernel_size, strides=pool_size,
                            name='spp_deconv', reuse=reuse, reg=reg)
        print('deconv_output:' + str(bottom.shape))
    return bottom
