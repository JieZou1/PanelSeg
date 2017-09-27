import cv2
from keras.engine import Model
from keras.models import load_model


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


if __name__ == "__main__":
    test_label_non_label_model()
    pass
