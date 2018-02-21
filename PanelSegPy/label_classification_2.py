import os
import tensorflow as tf

DATA_DIRECTORY = 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\'
MODEL_DIRECTORY = 'Z:\\Users\\jie\\projects\\PanelSeg\\ExpPython\\models\\label_classification_2\\'


def dataset(label_folder, non_label_folder):
    # Read filenames
    all_label_files, all_non_label_files = [], []
    folders = [dI for dI in os.listdir(label_folder) if os.path.isdir(os.path.join(label_folder, dI))]
    for f in folders:
        folder = os.path.join(label_folder, f)
        src_file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.png')]
        all_label_files += src_file

    all_non_label_files = [os.path.join(non_label_folder, file) for file in os.listdir(non_label_folder) if file.endswith('.png')]

    filenames = all_label_files + all_non_label_files
    labels = [0]*len(all_label_files) + [1]*len(all_non_label_files)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    pass



def main(argv):
    dataset(DATA_DIRECTORY+'Labels-28', DATA_DIRECTORY+'NonLabels-28')
    # test_input()
    # test_classification(argv)
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
