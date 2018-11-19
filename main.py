from __future__ import print_function
import argparse
import os
import tensorflow as tf
import numpy as np
import time
import math
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default = './')
parser.add_argument("--images_path", type = str)
parser.add_argument("--seg_path", type = str,default = None)
parser.add_argument("--depth_path", type = str,default = None)
parser.add_argument("--input_shape", type=int , default = (448,128))
    
parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--val_ratio", type = float, default = 0.1 )
parser.add_argument("--retrain", type = bool, default = False )
parser.add_argument("--predict", type = bool, default = False )
parser.add_argument("--batch_size", type = int, default = 2 )
    
args = parser.parse_args()
    
images_path = args.images_path
seg_path = args.seg_path
depth_path = args.depth_path
batch_size = args.batch_size
input_shape = args.input_shape
val_ratio = args.val_ratio
pre_flag = args.predict

save_weights_path = os.path.join(args.save_weights_path, 'tf_seg_depth_model')
output_path = args.save_weights_path
epochs = args.epochs
retrain = args.retrain
    

def main():
    val_data = None
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        #create model
        model = Seg_Depth_Model(sess, height=input_shape[1], weight=input_shape[0], batch_size = batch_size)
        if not pre_flag:
            if val_ratio:
                train_data,val_data = train_data_iterator(images_path,images_path,seg_path,depth_path,img_shape =input_shape,
                                             batch_size=batch_size,val_ratio = val_ratio,shuffle=True,augment = True)
            else:
                train_data = train_data_iterator(images_path,images_path,seg_path,depth_path,img_shape =input_shape,
                                                 batch_size=batch_size,val_ratio = val_ratio,shuffle=True,augment = True)
        
            model.train(train_data,val_data = val_data,learning_rate = 0.005, epochs = 5,steps_per_epoch = 25,save_path,retrain = retrain)
        else:
            img_data = predict_data_iterator(left_images_path,right_images_path,img_shape = input_shape,batch_size=1)
            seg_imgs,depth_imgs = model.predict(img_data,load_path = save_weights_path)
            for i,(seg,depth) in enumerate(zip(seg_imgs,depth_imgs)):
                tf.io.write_file(os.path.join(output_path,'seg'+str(i)+'.png'),seg)
                tf.io.write_file(os.path.join(output_path,'depth'+str(i)+'.png'),depth)

            
if __name__ == '__main__':
   main()
