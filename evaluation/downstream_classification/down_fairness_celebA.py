'''
Evaluate the utility/fairness of the generated data using downstream classification task.

Codebase:
    - G-PATE: https://github.com/AI-secure/G-PATE.git
'''


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import os
import json
import joblib
from tqdm import tqdm
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from collections import defaultdict


parser = argparse.ArgumentParser(description='Train classifier and evaluate their accuracy')
parser.add_argument('--data', type=str, help='datafile name')
parser.add_argument('--img_size', type=int, default=32, help="img size")
parser.add_argument('--attr', type=str, default='gender', help="lable attribute")
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--skip_train', action='store_true', default=False, help="only evaluate the model")
parser.add_argument('--gpu', type=str, default='0', help="gpu number")
parser.add_argument('--num_train', type=int, default=3, help="train number")






args = parser.parse_args()
savedir = args.data.split('/')[-2]
os.makedirs(savedir, exist_ok=True)
checkpoint_filepath = f'./{savedir}/best_model.h5'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

try:
    data = joblib.load(args.data)

except:
    data = np.zeros((100000, 12290))
    dim = 0

    for i in tqdm(range(args.range)):
        x =  joblib.load(args.data + f'-{i}.pkl')
        data[dim: dim+len(x)] = x
        dim += len(x)
    
    raise ValueError("Data not found")

print("Loaded generated data: ")
print("path: ", args.data)
print("shape: ", data.shape)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config));
tf.set_random_seed(args.seed)




def load_celeb():
    celebA_directory = './dataset/celebA/'
    tst_x = np.load(celebA_directory + f'test_celeba_{args.attr}_{args.img_size}x{args.img_size}.npz')['data_x']
    tst_y = np.load(celebA_directory + f'test_celeba_{args.attr}_{args.img_size}x{args.img_size}.npz')['data_y']
    print("Loaded test data: ")
    print(tst_y.sum(), len(tst_y))  

    # construct test data
    balance_num = int(min(tst_y.sum(), len(tst_y) - tst_y.sum()))
    biased_num = balance_num // 2
    indices = np.concatenate([np.where(tst_y == 1)[0][:balance_num], np.where(tst_y == 0)[0][:biased_num]])
    tst_x = tst_x[indices]
    tst_y = tst_y[indices]
    tst_y = np_utils.to_categorical(tst_y, 2)

    return tst_x, tst_y



x_test, y_test = load_celeb()



def pipeline(data):
    print(data.shape)
    x, label = np.hsplit(data, [-2])
    nb_classes = 2
    label = label.reshape((label.shape[0], nb_classes),order='F')
    x = x.reshape(x.shape[0], args.img_size, args.img_size, 3)




    model = Sequential()
    model.add(Conv2D(args.img_size //2 , kernel_size=3, activation='relu', input_shape=(args.img_size, args.img_size, 3), name='Conv2D-1'))
    model.add(MaxPooling2D(pool_size=2, name='MaxPool'))
    model.add(Dropout(0.2, name='Dropout-1'))
    model.add(Conv2D(args.img_size, kernel_size=3, activation='relu', name='Conv2D-2'))
    model.add(Dropout(0.25, name='Dropout-2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(args.img_size, activation='relu', name='Dense'))
    model.add(Dense(nb_classes, activation='softmax', name='Output'))
    sgd = optimizers.sgd(lr=1e-4) #, decay=1e-6, momentum=0.9, nesterov=True)


    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    print(x.shape)
    print(label.shape)
    print(x_test.shape)
    print(y_test.shape)


    # early stopping callback
    early_stopping = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 3, mode = 'auto')


    # ModelCheckpoint callback
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, 
                                    save_best_only=True, 
                                    monitor='val_accuracy', 
                                    mode='max', 
                                    verbose=1)


    # fit model with callbacks
    model.fit(x, label, batch_size=512, epochs=200, \
                        validation_data=(x_test, y_test), \
                        shuffle=True, \
                        callbacks = [model_checkpoint, early_stopping])
                


result = defaultdict(list)
if not args.skip_train:
    for i in range(args.num_train):
        print(f"==== {i+1}th Training =====")
        pipeline(data)


        # Load the saved model and evaluate
        model = load_model(checkpoint_filepath)
        y_pred = model.predict(x_test)
        y_test_label = np.argmax(y_test, axis=1)


        # compute overall accuracy
        overall_accuracy = sum(y_test_label == np.argmax(y_pred, axis=1)) / len(y_test_label)
        result['overall_accuracy'].append(overall_accuracy)
        print("Max acc:", result['overall_accuracy'])


        # Compute class-wise accuracy
        for class_id in range(2):
            class_indices = (y_test_label == class_id)
            class_accuracy = sum(y_test_label[class_indices] == np.argmax(y_pred[class_indices], axis=1)) / len(y_test_label[class_indices])
            result[f'y_{class_id}_accuracy'].append(class_accuracy)
            print(f"Class {class_id} Accuracy: {class_accuracy:.3f}")


# save result
with open(f"./{savedir}/result.txt", "w") as f:
    f.write(json.dumps(result, indent=4, sort_keys=True))

    # mean, std
    f.write("\n\n")
    f.write("Overall:\n")
    f.write(f"\tMean: {np.mean(result['overall_accuracy']):.3f}\n")
    f.write(f"\tStd: {np.std(result['overall_accuracy']):.3f}\n")

    for class_id in range(2):
        f.write(f"Class {class_id}:\n")
        f.write(f"\tMean: {np.mean(result[f'y_{class_id}_accuracy']):.3f}\n")
        f.write(f"\tStd: {np.std(result[f'y_{class_id}_accuracy']):.3f}\n")

