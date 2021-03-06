import argparse
import random
from collections import OrderedDict

import cv2
import keras
import os
import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.models import load_model
from keras.optimizers import RMSprop

import nn_cnn_3_layer
from Panel import Panel
from Config import Config


def cnn_feature_extraction(figure, rois, model_classifier):
    # Not working yet
    patches = np.empty([rois.shape[0], 28, 28, 3], dtype=np.uint8)
    for idx, roi in enumerate(rois):
        x, y, w, h = roi[0], roi[1], roi[2], roi[3]
        patch = figure.image[y:y + h, x:x + w]
        patches[idx] = cv2.resize(patch, (28, 28))

    patches = patches.astype('float32')
    patches[:, :, :, 0] -= 128.0  # c.img_channel_mean[0]
    patches[:, :, :, 1] -= 128.0  # c.img_channel_mean[1]
    patches[:, :, :, 2] -= 128.0  # c.img_channel_mean[2]
    patches /= 255

    prediction = model_classifier.predict_proba(patches)
    x = model_classifier.layers[5].output
    return x


def load_train_validation_data(c):

    label_folder = c.labels_normalized_folder
    non_label_folder = c.nonlabels_normalized_folder

    panel_label_ordered_dict = OrderedDict(sorted(Panel.LABEL_FOLDER_MAPPING.items()))

    files_all = []  # files of all sample images

    # label samples
    for k, v in panel_label_ordered_dict.items():
        files = []
        for sub_folder in v:
            folder = os.path.join(label_folder, sub_folder)
            files += [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]

        if len(files) > 5000:  # we keep at most 5000 sample per class
            random.shuffle(files)
            files = files[:5000]
        files_all.append(files)

    # bg samples
    files_bg = [os.path.join(non_label_folder, file) for file in os.listdir(non_label_folder) if file.endswith('.png')]
    random.shuffle(files_bg)
    files_all.append(files_bg[:10000])

    # Randomly select 90% for training and 10% for validation
    train_files = []
    validation_files = []
    for files in files_all:
        random.shuffle(files)
        k = int(len(files) * 0.9)
        train_files.append(files[:k])
        validation_files.append(files[k:])

    for i, key in enumerate(panel_label_ordered_dict.keys()):
        print('training label {0} samples: {1}\n'.format(key, len(train_files[i])))
        print('validation label {0} samples: {1}\n'.format(key, len(validation_files[i])))

    print('training label {0} samples: {1}\n'.format('bg', len(train_files[-1])))
    print('validation label {0} samples: {1}\n'.format('bg', len(validation_files[-1])))

    x_train_all = np.array([])
    y_train_all = np.array([])
    x_test_all = np.array([])
    y_test_all = np.array([])

    for i, files in enumerate(validation_files):
        if c.color_type == cv2.IMREAD_COLOR:
            x_test = np.empty([len(files), 28, 28, 3], dtype=int)
        else:
            x_test = np.empty([len(files), 28, 28, 1], dtype=int)
        y_test = np.empty([len(files), 1], dtype=int)
        for k, file in enumerate(files):
            img = cv2.imread(file, c.color_type)
            if c.color_type == cv2.IMREAD_COLOR:
                img = img.reshape(28, 28, 3)
            else:
                img = img.reshape(28, 28, 1)
            x_test[k] = img
            y_test[k] = i
        if i == 0:
            x_test_all = x_test
            y_test_all = y_test
        else:
            x_test_all = np.vstack((x_test_all, x_test))
            y_test_all = np.vstack((y_test_all, y_test))

    for i, files in enumerate(train_files):
        if c.color_type == cv2.IMREAD_COLOR:
            x_train = np.empty([len(files), 28, 28, 3], dtype=int)
        else:
            x_train = np.empty([len(files), 28, 28, 1], dtype=int)

        y_train = np.empty([len(files), 1], dtype=int)
        for k, file in enumerate(files):
            img = cv2.imread(file, c.color_type)
            if c.color_type == cv2.IMREAD_COLOR:
                img = img.reshape(28, 28, 3)
            else:
                img = img.reshape(28, 28, 1)
            x_train[k] = img
            y_train[k] = i
        if i == 0:
            x_train_all = x_train
            y_train_all = y_train
        else:
            x_train_all = np.vstack((x_train_all, x_train))
            y_train_all = np.vstack((y_train_all, y_train))

    return x_train_all, y_train_all, x_test_all, y_test_all


def train_label_classification(c, model_file=None):

    label_folder = c.labels_normalized_folder
    non_label_folder = c.nonlabels_normalized_folder

    print('train_label_classification with label_folder={0} and non_label_folder={1}'.format(label_folder, non_label_folder))
    input("Press Enter to continue...")

    #  Build or load model
    if model_file is None:
        # create model
        if c.color_type == cv2.IMREAD_COLOR:
            img_input = Input(shape=(28, 28, 3))
        else:
            img_input = Input(shape=(28, 28, 1))

        # prediction = model_cnn_2_layer.nn_classify_50_plus_bg(img_input)
        prediction = nn_cnn_3_layer.nn_classify_50_plus_bg(img_input)
        # prediction = model_cnn_3_layer_2.nn_classify_50_plus_bg(img_input)
        model = Model(inputs=img_input, outputs=prediction)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    else:
        model = load_model(model_file)

    model.summary()

    # Load and normalize data
    x_train, y_train, x_test, y_test = load_train_validation_data(c)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    if c.color_type == cv2.IMREAD_COLOR:
        x_train[:, :, :, 0] -= c.img_channel_mean[0]
        x_train[:, :, :, 1] -= c.img_channel_mean[1]
        x_train[:, :, :, 2] -= c.img_channel_mean[2]
        x_test[:, :, :, 0] -= c.img_channel_mean[0]
        x_test[:, :, :, 1] -= c.img_channel_mean[1]
        x_test[:, :, :, 2] -= c.img_channel_mean[2]
    else:
        x_train -= c.img_pixel_mean
        x_test -= c.img_pixel_mean

    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # x_train.reshape(x_train.shape[0], 28, 28, 3)
    # x_test.reshape(x_test.shape[0], 28, 28, 3)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 51)
    y_test = keras.utils.to_categorical(y_test, 51)

    # Checkpointing is to save the network weights only when there is an improvement in classification accuracy
    # on the validation dataset (monitor=’val_acc’ and mode=’max’).
    file_path = "weights-improvement-{epoch:04d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=90,
              verbose=1,
              callbacks=callbacks_list,
              validation_data=(x_test, y_test)
              )
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('final_model.h5')
    # model.save_weights('final_model_weights.h5')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train DL network for Label (50 class + 1 bg) classification')
    parser.add_argument('op',
                        help='an operation to be conducted',
                        type=str,
                        choices=[
                            'train_label_classification',
                                 ]
                        )

    args = parser.parse_args()

    configure = Config()

    if args.op == 'train_label_classification':
        train_label_classification(configure)
