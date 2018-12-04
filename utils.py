import tensorflow as tf
import tensorflow.contrib as tfc
import os
import numpy as np
import cv2 as cv


def load_jpeg(image_path, resize=(128, 448)):

    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, resize, align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return image


def img_preprocess(image, normalize=True, prepro=True):
    if prepro:
        # darkness improve
        sd = np.floor(255 / np.log2(255))
        image = (sd * np.log2(image + 0.0001)).astype(np.uint8)

        # gamma enhance contrast
        sg = np.floor(np.power(255, 2) / 255)
        image = np.power(image, 2) / sg

    # normalize image
    if normalize:
        # image = image - np.mean(image)
        image = np.around((image - np.mean(image)) / np.std(image), decimals=3)

    return image


def multiclass_label_normalize(image, normalize=False):
    # ratio gray and scale by 12
    image = ((image[:,:,0]*1/3+image[:,:,1]/4+image[:,:,2]*5/12)/12).astype(np.uint8)
    # normalize image
    if normalize:
        sd = np.floor(255 / np.log2(255))
        image = (sd * np.log2(image + 0.0001)).astype(np.uint8)
    return image

def reg_label_normalize(image, normalize=False):
    # gray image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # normalize image
    if normalize:
        image = np.around((np.log2(image + 0.0001)), decimals=2)

    return image


def cvload_img(image_path, resize=(32, 112), temp_ratio=1.5):
    image = cv.imread(image_path)
    image = cv.resize(image, (np.int(resize[1]*temp_ratio), np.int(resize[0]*temp_ratio)),
                      interpolation=cv.INTER_LINEAR)
    return image

def cvload_img_val(image_path, resize=(32, 112)):
    image = cv.imread(image_path)
    image = cv.resize(image, (resize[1], resize[0]), interpolation=cv.INTER_LINEAR)
    return image


def img_path_array(path, val_ratio=0., pattern=''):
    if pattern == '':
        path_array = np.array([os.path.join(path, x) for x in os.listdir(path)])
        if val_ratio:
            train_path_array, val_path_array = train_val_split(path_array, val_ratio)
            return train_path_array, val_path_array
        else:
            return path_array
    else:
        path_array = np.array([os.path.join(path, x) for x in os.listdir(path) if os.path.splitext(x)[0][-1] == pattern])
        if val_ratio:
            train_path_array, val_path_array = train_val_split(path_array, val_ratio)
            return train_path_array, val_path_array
        else:
            return path_array


def train_val_split(images_path_array, val_ratio=0.1):
    num_of_samples = len(images_path_array)
    val_rat = np.int(val_ratio*num_of_samples)
    val_img_array = np.array(images_path_array[:val_rat])
    train_img_array = np.array(images_path_array[val_rat:])
    return train_img_array, val_img_array


def load_batch_img(images_path_array, random_index=None, img_shape=(32, 112), ran_aug=None, ran_sel=0.1, label=0,
                   normalize=True, prepro=True):
    if label == 1:
        if ran_aug is not None:
            batch_img = np.array(list(map(lambda x: augment(multiclass_label_normalize(cvload_img(x, img_shape), normalize),
                                                            ran_aug, ran_sel), images_path_array[random_index])))
        else:
            if random_index is None:
                batch_img = np.array(list(map(lambda x: multiclass_label_normalize(cvload_img_val(x, img_shape), normalize),
                                              images_path_array)))
            else:
                batch_img = np.array(list(map(lambda x: multiclass_label_normalize(cvload_img_val(x, img_shape), normalize),
                                              images_path_array[random_index])))
    elif label == 2:
        if ran_aug is not None:
            batch_img = np.array(list(map(lambda x: augment(reg_label_normalize(cvload_img(x, img_shape), normalize),
                                                            ran_aug, ran_sel), images_path_array[random_index])))
        else:
            if random_index is None:
                batch_img = np.array(list(map(lambda x: reg_label_normalize(cvload_img_val(x, img_shape), normalize),
                                              images_path_array)))
            else:
                batch_img = np.array(list(map(lambda x: reg_label_normalize(cvload_img_val(x, img_shape), normalize),
                                              images_path_array[random_index])))
    else:
        if ran_aug is not None:
            batch_img = np.array(list(map(lambda x: augment(img_preprocess(cvload_img(x, img_shape), normalize, prepro),
                                                            ran_aug, ran_sel), images_path_array[random_index])))
        else:
            if random_index is None:
                batch_img = np.array(list(map(lambda x: img_preprocess(cvload_img_val(x, img_shape), normalize, prepro),
                                              images_path_array)))
            else:
                batch_img = np.array(list(map(lambda x: img_preprocess(cvload_img_val(x, img_shape), normalize, prepro),
                                              images_path_array[random_index])))
    return batch_img


def augment(input_img, ran_degree, ran_aug,
            angle=30,
            img_size=(64, 224),
            temp_ratio=1.5):
    # width_shift_range=0.5
    # height_shift_range=0.2
    onehalf_size = (np.int(img_size[0] * temp_ratio), np.int(img_size[1] * temp_ratio))
    wshift_range = onehalf_size[1]-img_size[1]
    hshift_range = onehalf_size[0]-img_size[0]
    np_ran = ran_degree*2-1
    # half_size = (img_size[1]//2, img_size[0]//2)
    # quar_size = (onehalf_size[1]//4, onehalf_size[0]//4)
    # eith_size = (onehalf_size[1]//8, onehalf_size[0]//8)
    '''
    # perspective_transform
    pts1 = np.float32([[quar_size[0], quar_size[1]], [3 * quar_size[0], quar_size[1]],
                       [quar_size[0], 3 * quar_size[1]], [3 * quar_size[0], 3 * quar_size[1]]])
    pts2 = np.float32([[quar_size[0] - np_ran * eith_size[0], quar_size[1] - np_ran * eith_size[1]],
                       [3 * quar_size[0] + np_ran * eith_size[0], quar_size[1] - np_ran * eith_size[1]],
                       [quar_size[0] - np_ran * eith_size[0], 3 * quar_size[1] + np_ran * eith_size[1]],
                       [3 * quar_size[0] + np_ran * eith_size[0], 3 * quar_size[1] + np_ran * eith_size[1]]])
    M_perspective = cv.getPerspectiveTransform(pts1, pts2)
    # input_img = cv.warpPerspective(input_img, M_perspective, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
    input_img = cv.warpPerspective(input_img, M_perspective, (img_size[1], img_size[0]))
    '''

    if ran_aug < 0.25:
        # scale nad rotate by random -30~30 degree
        M_rot = cv.getRotationMatrix2D((onehalf_size[1]//2, onehalf_size[0]//2), angle*np_ran, 1)
        # input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
        input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]))
    elif ran_aug < 0.5:
        # scale nad rotate by random 150~210 degree
        M_rot = cv.getRotationMatrix2D((onehalf_size[1]//2, onehalf_size[0]//2), 180+angle * np_ran, 1)
        # input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]), borderMode=cv.BORDER_REFLECT)
        input_img = cv.warpAffine(input_img, M_rot, (img_size[1], img_size[0]))
    elif ran_aug < 0.75:
        # random_crop
        ran2 = ran_aug*4-2
        ind_h = np.int(hshift_range * ran_degree)
        ind_w = np.int(wshift_range * ran2)
        input_img = input_img[ind_h:img_size[0]+ind_h, ind_w:img_size[1]+ind_w]
    else:
        input_img = cv.resize(input_img, (img_size[1], img_size[0]), interpolation=cv.INTER_LINEAR)
        input_img = np.fliplr(input_img)
    '''
    ran_degree = ran_degree * 0.3 + 0.7
    # random lower brightness
    input_img = (input_img * ran_degree).astype(np.uint8)
    '''
    return input_img


def conv_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=-1, name=None, reuse=None, reg=1e-4,
               apply_bn=True, apply_elu=True):
    with tf.variable_scope(name):
        conv_params = {
            'padding': 'same',
            'kernel_initializer': tfc.layers.xavier_initializer(),
            'kernel_regularizer': tfc.layers.l2_regularizer(reg),
            'bias_regularizer': tfc.layers.l2_regularizer(reg),
            'name': 'conv',
            'reuse': reuse
        }
        if dilation_rate > -1:
            conv_params['dilation_rate'] = dilation_rate
        # if dilation_rate == -1:
        #    conv_params[]
        bottom = func(bottom, filters, kernel_size, strides, **conv_params)
        if apply_bn:
            bottom = tf.layers.batch_normalization(bottom,
                                                   training=tf.get_default_graph().get_tensor_by_name('is_training:0'),
                                                   reuse=reuse, name='bn')
        if apply_elu:
            bottom = tf.nn.elu(bottom, name='elu')
        return bottom


def res_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
              projection=True):
    with tf.variable_scope(name):
        short_cut = bottom
        bottom = tf.nn.elu(bottom, name='elu')
        bottom = conv_block(func, bottom, filters, 1, strides, dilation_rate, name='conv1', reuse=reuse,
                            reg=reg)
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, name='conv2', reuse=reuse,
                            reg=reg, apply_elu=False)
        bottom = conv_block(func, bottom, filters, 1, strides, dilation_rate, name='conv3', reuse=reuse,
                            reg=reg)
        if projection:
            short_cut = conv_block(func, short_cut, filters, 1, strides, dilation_rate, name='sconv3',
                                   reuse=reuse, reg=reg)
            short_cut = tf.nn.elu(short_cut, name='elu')
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
