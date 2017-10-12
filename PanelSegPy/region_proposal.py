import cv2
from keras import Input
from keras.engine import Model
from keras.models import load_model
from keras.optimizers import Adam

import model_cnn_3_layer

base_net_path = 'Z:\\Users\\jie\\projects\\PanelSeg\\Exp1\\models\\label50+bg_model-cnn_3_layer-0.9915.h5'


def test_label_non_label_model():
    model_file = '/Users/jie/projects/PanelSeg/Exp1/models/label_non_label_2_cnn_model.h5'
    image_file = '/Users/jie/projects/PanelSeg/data/0/1465-9921-6-6-4.jpg'

    model = load_model(model_file)
    model.summary()

    model_cnn_only = Model(inputs=model.input, outputs=model.get_layer('conv2d_2').output)
    model_cnn_only.summary()

    # load an image
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img = img.reshape((1, ) + img.shape + (1,))
    img = img.astype('float32')
    img /= 255

    # filter the image (feature extraction)
    prediction = model_cnn_only.predict(img)

    pass


def train_rpn():
    nn = model_cnn_3_layer

    input_shape_img = (None, None, 1)
    img_input = Input(shape=input_shape_img)
    rpn = nn.nn_rpn(img_input)

    # model_base = load_model(base_net_path)
    # model_base.summary()

    # create rpn model and load weights
    model_rpn = Model(img_input, rpn[:2])
    print('loading weights from {}'.format(base_net_path))
    model_rpn.load_weights(base_net_path, by_name=True)
    model_rpn.summary()

    optimizer = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])


if __name__ == "__main__":
    # test_label_non_label_model()
    train_rpn()
    pass
