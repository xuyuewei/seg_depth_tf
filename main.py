from __future__ import print_function
import argparse
import os
import tensorflow as tf
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default = './')
parser.add_argument("--images_path", type = str)
parser.add_argument("--seg_path", type = str,default = None)
parser.add_argument("--depth_path", type = str,default = None)
parser.add_argument("--input_shape", type=int , default = (128,448))
    
parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--val_ratio", type = float, default = 0.1 )
parser.add_argument("--predict", type = bool, default = False )
parser.add_argument("--finetune", type = bool, default = False )
parser.add_argument("--retrain", type = bool, default = True )
parser.add_argument("--shuffle", type = bool, default = True )
parser.add_argument("--augment", type = bool, default = True )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--learning_rate", type = float, default = 0.005 )
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
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #create model
        seg_depth = Seg_Depth_Model(sess,input_shape[0],input_shape[1], batch_size)
        if not pre_flag:
            seg_depth.train(train_data_path = [images_path,images_path,seg_path,depth_path],save_weights_path,batch_size,val_ratio,learning_rate, epochs,)
            seg_depth.train(train_data_path = [images_path,images_path,seg_path,depth_path],save_weights_path,batch_size,val_ratio,learning_rate/5, epochs,retrain)
        elif finetune:
            seg_depth.finetune(train_data_path = [images_path,images_path,seg_path,depth_path],save_weights_path,batch_size,val_ratio,learning_rate/5, epochs,retrain)
        else:
            img_data = predict_data_iterator(left_images_path,right_images_path,input_shape,batch_size)
            seg_imgs,depth_imgs = model.predict(img_data,save_weights_path)
            for i,(seg,depth) in enumerate(zip(seg_imgs,depth_imgs)):
                tf.io.write_file(os.path.join(output_path,'seg'+str(i)+'.png'),seg)
                tf.io.write_file(os.path.join(output_path,'depth'+str(i)+'.png'),depth)

if __name__ == '__main__':
   main()
