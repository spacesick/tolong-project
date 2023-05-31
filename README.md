# C23-PC725 Capstone Project

## Members

- (MD) A346DKX4282 - Fernandico Geovardo
- (ML) M049DSX0588 - Dhuha Ardha Saputra
- (ML) M181DSX3641 - Vincent Yovian
- (CC) C368DSX2892 - I Ketut Teguh Wibawa Lessmana P. T.
- (CC) C220DSY0626 - Audy Revi Nugraha
- (CC) C303DKY3970 - Vanessa Evlin


## Machine learning model

We create a deep neural network for classifying seven types of injuries that often happen in accidents.

### Dataset info

We collected our Dataset various wound type from kaggle. We got 431 wound type image data in total and there is the distribution.

Abrasions: 85
Bruises: 122
Burns: 59
Cuts: 50
Ingrown Nails: 31
Lacerations: 61
Stab Wounds: 23

### Architecture

We use the technique of transfer learning by adding multiple layers of fully connected networks on top of an InceptionV3 model pre-trained on the imagenet dataset. We freeze all the layers in the pre-trained InceptionV3 except for the last 12. We saved our best performing model in .h5 format and its weights in a .ckpt file, which so far has reached a validation accuracy of 0.8315. The latest checkpoints can be found [here](https://drive.google.com/drive/folders/1PHNyZyMKG6q6ibdMQEZ6wYGbwXP243yX?usp=sharing).

## Cloud computing

TODO

## Mobile Development
   
TODO
