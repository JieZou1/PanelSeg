import keras
import numpy as np
import random
import shutil
import cv2
import os
import pandas as pd
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import load_model
from keras.optimizers import RMSprop

from Figure import Figure


def test_load_image():
    image_file = "/Users/jie/projects/PanelSeg/data/0/1465-9921-6-6-4.jpg"
    figure = Figure(image_file)
    figure.load_image()
    cv2.imshow("orig", figure.image_orig)
    cv2.imshow("padded", figure.image)
    cv2.imshow("gray", figure.image_gray)
    cv2.waitKey()


def test_crop_label_patches():
    image_file = "/Users/jie/projects/PanelSeg/data/0/1465-9921-6-6-4.jpg"
    figure = Figure(image_file)
    figure.load_image()
    figure.load_gt_annotation(which_annotation='label')
    label_patches = figure.crop_label_patches(is_gray=True)
    for i in range(0, len(label_patches)):
        patch = label_patches[i]
        filename = "Patch{}.jpg".format(i)
        cv2.imwrite(filename, patch)


def generate_statistics(list_file="/Users/jie/projects/PanelSeg/Exp1/all.txt"):
    """
    Generate label statistics for deciding some parameters of the algorithm
    :param list_file:
    :return:
    """
    with open(list_file) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    for figure in figures:
        figure.load_gt_annotation(which_annotation='label')
        figure.load_image()

    # Figure Image Statistics
    image_width, image_height = [], []
    for figure in figures:
        height, width = figure.image_orig.shape[:2]
        image_width.append(width)
        image_height.append(height)

    print('\nimage width statistics:')
    print(pd.Series(image_width).describe())
    print('\nimage height statistics:')
    print(pd.Series(image_height).describe())

    # Label Statistics
    label_width, label_height = [], []
    for figure in figures:
        for panel in figure.panels:
            width, height = panel.label_rect[2:]
            label_width.append(width)
            label_height.append(height)

    print('\nLabel width statistics:')
    print(pd.Series(label_width).describe())
    print('\nLabel height statistics:')
    print(pd.Series(label_height).describe())


def generate_label_train_samples(list_file="/Users/jie/projects/PanelSeg/Exp1/all.txt",
                                 target_folder="/Users/jie/projects/PanelSeg/Exp1/Labels"):
    """
    Generate label train sample (image patches).
    :param list_file:
    :param target_folder:
    :return:
    """
    with open(list_file) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    # Clear the folder
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.mkdir(target_folder)

    for figure in figures:
        print("Processing {0}".format(figure.id))
        figure.load_gt_annotation(which_annotation='label')
        figure.load_image()
        figure.crop_label_patches(is_gray=True)
        figure.save_label_patches("/Users/jie/projects/PanelSeg/Exp1/Labels")


def normalize_label_train_samples(src_folder='/Users/jie/projects/PanelSeg/Exp1/Labels',
                                  dst_folder='/Users/jie/projects/PanelSeg/Exp1/Labels-28'):
    # Collect only single char label only for now
    folders = [dI for dI in os.listdir(src_folder) if (',' not in dI) and os.path.isdir(os.path.join(src_folder, dI))]

    src_files = []
    for f in folders:
        folder = os.path.join(src_folder, f)
        src_file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        src_files.append(src_file)

    # print(pd.Series([len(k) for k in src_files]).describe())
    # select 5,000 if more than 5,000
    for i in range(len(src_files)):
        src_file = src_files[i]
        if len(src_file) > 5000:
            src_files[i] = random.sample(src_file, 5000)

    for src_file in src_files:
        for file in src_file:
            print("Processing {0}".format(file))
            src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            dst_img = cv2.resize(src_img, (28, 28))
            path = file.replace(src_folder, dst_folder)
            folder, file = os.path.split(path)
            if not os.path.exists(folder):
                os.mkdir(folder)
            cv2.imwrite(path, dst_img)


def generate_nonlabel_train_samples(list_file="/Users/jie/projects/PanelSeg/Exp1/all.txt",
                                    target_folder="/Users/jie/projects/PanelSeg/Exp1/NonLabels"):
    with open(list_file) as f:
        lines = f.readlines()

    # Remove whitespace characters, and then construct the figures
    figures = [Figure(line.strip()) for line in lines]

    # Clear the folder
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    for figure in figures:
        print("Processing {0}".format(figure.id))
        figure.load_image()
        image_height, image_width = figure.image_orig.shape[:2]
        for i in range(50):
            x, y, s = random.randint(-5, image_width-5), random.randint(-5, image_height-5), round(random.gauss(20, 7))
            if (s < 5) or (s > 80):
                continue
            if (x + s - image_width > s / 2) or (0-x > s/2):
                continue
            if (y + s - image_height > s / 2) or (0-y > s/2):
                continue

            rect = (x, y, s, s)
            patch_file = figure.id + "_".join(str(x) for x in rect) + ".png"

            x += figure.PADDING
            y += figure.PADDING
            patch = figure.image_gray[y:y+s, x:x+s]

            patch_file = os.path.join(target_folder, patch_file)
            cv2.imwrite(patch_file, patch)


def normalize_nonlabel_train_samples(src_folder='/Users/jie/projects/PanelSeg/Exp1/NonLabels',
                                     dst_folder='/Users/jie/projects/PanelSeg/Exp1/NonLabels-28'):
    src_files = [os.path.join(src_folder, file) for file in os.listdir(src_folder) if file.endswith('.png')]
    src_files = random.sample(src_files, 30000)
    # normalize images to 28x28
    for file in src_files:
        print("Processing {0}".format(file))
        src_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        dst_img = cv2.resize(src_img, (28, 28))
        path = file.replace(src_folder, dst_folder)
        folder, file = os.path.split(path)
        if not os.path.exists(folder):
            os.mkdir(folder)
        if not os.path.exists(path):
            cv2.imwrite(path, dst_img)


def load_train_validation_data(label_folder='/Users/jie/projects/PanelSeg/Exp1/Labels-28',
                               non_label_folder='/Users/jie/projects/PanelSeg/Exp1/NonLabels-28'):
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

    train_label_files = all_label_files[:100000]
    train_non_label_files = all_non_label_files[:100000]

    validation_label_files = all_label_files[100000]
    validation_non_label_files = all_non_label_files[100000]

    print('training label samples: {0}\n'.format(len(train_label_files)))
    print('training nonlabel samples: {0}\n'.format(len(train_non_label_files)))
    print('validation label samples: {0}\n'.format(len(validation_label_files)))
    print('validation nonlabel samples: {0}\n'.format(len(validation_non_label_files)))

    n_train_samples = len(train_label_files) + len(train_non_label_files)
    n_test_samples = len(validation_label_files) + len(validation_non_label_files)

    x_train = np.empty([n_train_samples, 28, 28, 1], dtype=int)
    y_train = np.empty([n_train_samples, 1], dtype=int)
    x_test = np.empty([n_test_samples, 28, 28, 1], dtype=int)
    y_test = np.empty([n_test_samples, 1], dtype=int)

    for i in range(len(train_label_files)):
        file = train_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28, 28, 1)
        x_train[i] = img
        y_train[i] = 0

    for i in range(len(train_non_label_files)):
        file = train_non_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28, 28, 1)
        x_train[len(train_label_files) + i] = img
        y_train[len(train_label_files) + i] = 1

    for i in range(len(validation_label_files)):
        file = validation_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28, 28, 1)
        x_test[i] = img
        y_test[i] = 0

    for i in range(len(validation_non_label_files)):
        file = validation_non_label_files[i]
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(28, 28, 1)
        x_test[len(validation_label_files) + i] = img
        y_test[len(validation_label_files) + i] = 1

    return x_train, y_train, x_test, y_test


def train_label_none_label_classification_lenet5(model_file=None):

    # Load and normalize data
    x_train, y_train, x_test, y_test = load_train_validation_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test.reshape(x_test.shape[0], 28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    if model_file is None:
        # create model
        inputs = Input(shape=(28, 28, 1))
        cov1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(cov1)
        cov2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(cov2)
        flat1 = Flatten()(pool2)
        dense1 = Dense(128, activation='relu')(flat1)
        predictions = Dense(2, activation='softmax')(dense1)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    else:
        model = load_model(model_file)

    model.summary()

    # Checkpointing is to save the network weights only when there is an improvement in classification accuracy
    # on the validation dataset (monitor=’val_acc’ and mode=’max’).
    file_path = "weights-improvement-{epoch:04d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
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
    # Test and Generate some statistics
    # test_load_image()
    # test_crop_label_patches()
    # generate_statistics()

    # Generate and Normalize samples
    # generate_label_train_samples()
    # normalize_label_train_samples()
    # generate_nonlabel_train_samples()
    # normalize_nonlabel_train_samples()

    # Training models
    train_label_none_label_classification_lenet5('final_model.h5')
    # train_label_none_label_classification_lenet5()
