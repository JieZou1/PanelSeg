import cv2

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


def generate_label_train_samples(list_file="/Users/jie/projects/PanelSeg/Exp1/Set0.txt",
                                 target_folder="/Users/jie/projects/PanelSeg/Exp1/Label"):
    """
    Generate label train sample (image patches).
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
        figure.crop_label_patches(is_gray=True)
        figure.save_label_patches("/Users/jie/projects/PanelSeg/Exp1/Labels")

    # print some statistics
    total_label = 0
    for figure in figures:
        total_label += len(figure.panels)
    print("Total number of labels is {}".format(total_label))


if __name__ == "__main__":
    # test_load_image()
    # test_crop_label_patches()
    generate_label_train_samples()
