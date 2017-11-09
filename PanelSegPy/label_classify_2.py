import argparse

import keras
import numpy as np
import random
import cv2
import os
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.models import load_model
from keras.optimizers import RMSprop

import nn_cnn_2_layer
import nn_cnn_3_layer
import nn_cnn_3_layer_2
from Config import Config
from Figure import Figure


def load_train_validation_data(label_folder, non_label_folder):
    all_label_files, all_non_label_files = [], []
    folders = [dI for dI in os.listdir(label_folder) if os.path.isdir(os.path.join(label_folder, dI))]
    for f in folders:
        folder = os.path.join(label_folder, f)
        src_file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        all_label_files += src_file

    all_non_label_files = [os.path.join(non_label_folder, file) for file in os.listdir(non_label_folder) if file.endswith('.png')]

    # Select 100,000 for training and the remaining for validation
    random.shuffle(all_label_files)
    random.shuffle(all_non_label_files)

    train_label_files = all_label_files[:90000]
    train_non_label_files = all_non_label_files[:90000]

    validation_label_files = all_label_files[90000:]
    validation_non_label_files = all_non_label_files[90000:]

    print('training label samples: {0}\n'.format(len(train_label_files)))
    print('training nonlabel samples: {0}\n'.format(len(train_non_label_files)))
    print('validation label samples: {0}\n'.format(len(validation_label_files)))
    print('validation nonlabel samples: {0}\n'.format(len(validation_non_label_files)))

    n_train_samples = len(train_label_files) + len(train_non_label_files)
    n_test_samples = len(validation_label_files) + len(validation_non_label_files)

    x_train = np.empty([n_train_samples, 28, 28, 3], dtype=int)
    y_train = np.empty([n_train_samples, 1], dtype=int)
    x_test = np.empty([n_test_samples, 28, 28, 3], dtype=int)
    y_test = np.empty([n_test_samples, 1], dtype=int)

    for i in range(len(train_label_files)):
        file = train_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = img.reshape(28, 28, 3)
        x_train[i] = img
        y_train[i] = 0

    for i in range(len(train_non_label_files)):
        file = train_non_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = img.reshape(28, 28, 3)
        x_train[len(train_label_files) + i] = img
        y_train[len(train_label_files) + i] = 1

    for i in range(len(validation_label_files)):
        file = validation_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = img.reshape(28, 28, 3)
        x_test[i] = img
        y_test[i] = 0

    for i in range(len(validation_non_label_files)):
        file = validation_non_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = img.reshape(28, 28, 3)
        x_test[len(validation_label_files) + i] = img
        y_test[len(validation_label_files) + i] = 1

    return x_train, y_train, x_test, y_test


def train_label_none_label_classification(label_folder, non_label_folder, model_file=None):

    c = Config()

    #  Build or load model
    if model_file is None:
        # create model
        img_input = Input(shape=(28, 28, 3))
        # prediction = model_cnn_2_layer.nn_classify_label_non_label(img_input)
        # prediction = model_cnn_3_layer.nn_classify_label_non_label(img_input)
        prediction = nn_cnn_3_layer.nn_classify_label_non_label(img_input)
        model = Model(inputs=img_input, outputs=prediction)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    else:
        model = load_model(model_file)

    model.summary()

    # Load and normalize data
    x_train, y_train, x_test, y_test = load_train_validation_data(label_folder, non_label_folder)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] -= c.img_channel_mean[0]
    x_train[:, :, :, 1] -= c.img_channel_mean[1]
    x_train[:, :, :, 2] -= c.img_channel_mean[2]
    x_test[:, :, :, 0] -= c.img_channel_mean[0]
    x_test[:, :, :, 1] -= c.img_channel_mean[1]
    x_test[:, :, :, 2] -= c.img_channel_mean[2]

    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # x_train.reshape(x_train.shape[0], 28, 28, 3)
    # x_test.reshape(x_test.shape[0], 28, 28, 3)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    # Checkpointing is to save the network weights only when there is an improvement in classification accuracy
    # on the validation dataset (monitor=’val_acc’ and mode=’max’).
    file_path = "weights-improvement-{epoch:04d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=100,
              verbose=1,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test)
              )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('final_model.h5')
    # model.save_weights('final_model_weights.h5')


def label_none_label_classification(label_folder, non_label_folder, model_file):
    model = load_model(model_file)
    model.summary()

    all_label_files, all_non_label_files = [], []
    folders = [dI for dI in os.listdir(label_folder) if os.path.isdir(os.path.join(label_folder, dI))]
    for f in folders:
        folder = os.path.join(label_folder, f)
        src_file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        all_label_files += src_file
    all_non_label_files = [os.path.join(non_label_folder, file) for file in os.listdir(non_label_folder) if file.endswith('.png')]

    n_label_samples = len(all_label_files)
    n_nonlabel_samples = len(all_non_label_files)

    print('Label samples: {0}\n'.format(n_label_samples))
    print('Nonlabel samples: {0}\n'.format(n_nonlabel_samples))

    label_samples = np.empty([n_label_samples, 28, 28, 1], dtype=int)
    non_label_samples = np.empty([n_nonlabel_samples, 28, 28, 1], dtype=int)

    for i, file in enumerate(all_label_files):
        file = all_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28, 28, 1)
        label_samples[i] = img

    label_samples = label_samples.astype('float32')
    label_samples /= 255

    prediction = model.predict(label_samples)
    idx = np.ix_(prediction[:, 0] < 0.5)
    label_errors = np.array(all_label_files)[idx]
    np.savetxt("label_error.txt", label_errors, fmt='%s', newline='\n')

    for i, file in enumerate(all_non_label_files):
        file = all_non_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28, 28, 1)
        non_label_samples[i] = img

    non_label_samples = non_label_samples.astype('float32')
    non_label_samples /= 255

    prediction = model.predict(non_label_samples)
    idx = np.ix_(prediction[:, 1] < 0.5)
    non_label_errors = np.array(all_non_label_files)[idx]
    np.savetxt("non_label_error.txt", non_label_errors, fmt='%s', newline='\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train DL network for Label Non-Label patch classification')
    parser.add_argument('op',
                        help='an operation to be conducted',
                        type=str,
                        choices=[
                            'train_label_none_label_classification',
                            'continue_train_label_none_label_classification',
                            'label_none_label_classification'
                                 ]
                        )
    parser.add_argument('--list_file',
                        help='the list file of the figure images',
                        type=str,
                        default='/Users/jie/projects/PanelSeg/ExpRcnn/all.txt')
    parser.add_argument('--src_folder',
                        help='the source folder to load data from',
                        type=str,
                        default='/Users/jie/projects/PanelSeg/ExpRcnn/Labels')
    parser.add_argument('--target_folder',
                        help='the target folder to save results',
                        type=str,
                        default='/Users/jie/projects/PanelSeg/ExpRcnn/Labels')
    parser.add_argument('--label_folder',
                        help='the label folder for training the network',
                        type=str,
                        default='/Users/jie/projects/PanelSeg/ExpRcnn/Labels-28')
    parser.add_argument('--non_label_folder',
                        help='the non-label folder for training the network',
                        type=str,
                        default='/Users/jie/projects/PanelSeg/ExpRcnn/NonLabels-28')
    parser.add_argument('--model_file',
                        help='the network model file to load from',
                        type=str,
                        default='/Users/jie/projects/PanelSeg/ExpRcnn/models/label_non_label_2_cnn_model.h5')

    args = parser.parse_args()

    if args.op == 'train_label_none_label_classification':
        print('train_label_none_label_classification with label_folder={0} and non_label_folder={1}'
              .format(args.label_folder, args.non_label_folder))
        input("Press Enter to continue...")
        train_label_none_label_classification(args.label_folder, args.non_label_folder)
        # train_label_none_label_classification_lenet5(label_folder="label_folder='/Users/jie/projects/PanelSeg/Exp1/Labels-28',
        #                        non_label_folder='/Users/jie/projects/PanelSeg/Exp1/NonLabels-28')

    elif args.op == 'continue_train_label_none_label_classification':
        print('continue_train_label_none_label_classification with label_folder={0}, non_label_folder={1} and model_file={2}'
              .format(args.label_folder, args.non_label_folder, args.model_file))
        input("Press Enter to continue...")
        train_label_none_label_classification(args.label_folder, args.non_label_folder, args.model_file)
        # train_label_none_label_classification(label_folder="label_folder='/Users/jie/projects/PanelSeg/Exp1/Labels-28',
        #                           non_label_folder='/Users/jie/projects/PanelSeg/Exp1/NonLabels-28',
        #                           model_file='/Users/jie/projects/PanelSeg/Exp1/models/label_non_label_2_cnn_model.h5')

    elif args.op == 'label_none_label_classification':
        print('label_none_label_classification with label_folder={0}, non_label_folder={1} and model_file={2}'
              .format(args.label_folder, args.non_label_folder, args.model_file))
        input("Press Enter to continue...")
        label_none_label_classification(args.label_folder, args.non_label_folder, args.model_file)
        # train_label_none_label_classification_lenet5(label_folder="label_folder='/Users/jie/projects/PanelSeg/Exp1/Labels-28',
        #                           non_label_folder='/Users/jie/projects/PanelSeg/Exp1/NonLabels-28',
        #                           model_file='/Users/jie/projects/PanelSeg/Exp1/models/label_non_label_2_cnn_model.h5')

    else:
        print('Operation {0} is not implemented yet!'.format(args.op))
