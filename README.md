# seg_depth_tf
### Segmentation and Depth_estimation


### Train
As an example, use the following command to train a seg_depth model on your dataset

```
python main.py --save_weights_path '/home/xdjf/xuyuewei' \
               --images_path '/home/xdjf/xuyuewei/training/image' \
               --seg_path '/home/xdjf/xuyuewei/training/semantic_rgb' \
               --depth_path '/home/xdjf/xuyuewei/training/disp' \
               --epochs 10 \
               --val_ratio 0.1 \
               --batch_size 5 \
               --learning_rate 0.005
```
As another example, use the following command to finetune a seg_depth

### Finetune
```
python main.py --save_weights_path '/home/xdjf/xuyuewei' \
               --images_path '/home/xdjf/xuyuewei/training/image' \
               --seg_path '/home/xdjf/xuyuewei/training/semantic_rgb' \
               --depth_path '/home/xdjf/xuyuewei/training/disp' \
               --epochs 10 \
               --val_ratio 0.1 \
               --batch_size 5 \
               --learning_rate 0.002 \
               --finetune True
```
### Predict
Use the following command to predict on test data

```
python main.py --save_weights_path '/home/xdjf/xuyuewei' \
               --images_path '' \
               --epochs 10 \
               --val_ratio 0.1 \
               --batch_size 5 \
               --predict True
```
