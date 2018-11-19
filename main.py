from __future__ import print_function
import argparse
import os
import tensorflow as tf
from model import *

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default = './')
parser.add_argument("--images_path", type = str)
parser.add_argument("--seg_path", type = str,default = None)
parser.add_argument("--depth_path", type = str,default = None)
parser.add_argument("--input_shape", type=int , default = (448,128))
    
parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--val_ratio", type = float, default = 0.1 )
parser.add_argument("--predict", type = bool, default = False )
parser.add_argument("--finetune", type = bool, default = False )
parser.add_argument("--retrain", type = bool, default = True )
parser.add_argument("--shuffle", type = bool, default = True )
parser.add_argument("--augment", type = bool, default = True )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--learning_rate", type = int, default = 0.005 )
args = parser.parse_args()
    
images_path = args.images_path
seg_path = args.seg_path
depth_path = args.depth_path
batch_size = args.batch_size
input_shape = args.input_shape
val_ratio = args.val_ratio
learning_rate = args.learning_rate
pre_flag = args.predict
finetune = args.finetune

save_weights_path = os.path.join(args.save_weights_path, 'tf_seg_depth_model')
output_path = args.save_weights_path
epochs = args.epochs

retrain = args.retrain
shuffle = args.shuffle
augment = args.augment
def main():
    val_data = None
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session() as sess:
        #create model
        model = Seg_Depth_Model(sess,input_shape[1],input_shape[0], batch_size)
        if not pre_flag:
            if val_ratio:
                train_data,val_data = train_data_iterator(images_path,images_path,seg_path,depth_path,input_shape,
                                                          batch_size,val_ratio,shuffle, augment)
            else:
                train_data = train_data_iterator(images_path,images_path,seg_path,depth_path,input_shape,
                                                 batch_size,val_ratio,shuffle,augment)
        if not pre_flag:
            model.build_model()
            model.train(train_data,save_weights_path,val_data,learning_rate, epochs,25)
            model.train(train_data,save_weights_path,val_data,learning_rate/5, epochs,25,retrain)
        elif finetune:
            model.finetune(train_data,save_weights_path,val_data,learning_rate, epochs,25)
        else:
            img_data = predict_data_iterator(left_images_path,right_images_path,input_shape,batch_size)
            seg_imgs,depth_imgs = model.predict(img_data,save_weights_path)
            for i,(seg,depth) in enumerate(zip(seg_imgs,depth_imgs)):
                tf.io.write_file(os.path.join(output_path,'seg'+str(i)+'.png'),seg)
                tf.io.write_file(os.path.join(output_path,'depth'+str(i)+'.png'),depth)

if __name__ == '__main__':
   main()
