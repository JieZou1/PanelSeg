import argparse
import sys
import os
import csv
from figure.figure_set import FigureSet
from figure.figure import Figure


def parse_args(args):
    parser = argparse.ArgumentParser(description='Convert PanelSeg data set to CSV format.')
    parser.add_argument('--list_path',
                        help='The path to the list file.',
                        default='\\Users\\jie\\projects\\PanelSeg\\ExpKeras\\all.txt',
                        type=str)
    parser.add_argument('--annotation_path',
                        help='The output annotation CSV file.',
                        default='\\Users\\jie\\projects\\PanelSeg\\ExpKeras\\panel_split\\all.csv',
                        type=str)
    # parser.add_argument('mapping_path',
    #                     help='The output class to ID mapping CSV file.',
    #                     default='Z:\\Users\\jie\\projects\\PanelSeg\\ExpKeras\\mapping.csv',
    #                     type=str)
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    figure_set = FigureSet()
    figure_set.load_list(args.list_path)
    with open(args.annotation_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for idx, file in enumerate(figure_set.files):
            # if '1465-9921-6-55-4' not in file:
            #     continue
            print('Processing Image {:d}: {:s}.'.format(idx, file))
            figure = Figure(file)
            figure.load_image()
            xml_path = os.path.join(figure.image_path.replace('.jpg', '_data.xml'))
            figure.load_annotation_iphotodraw(xml_path)

            # write to CSV file
            for panel in figure.panels:
                csv_writer.writerow([figure.image_path,
                                    str(panel.panel_rect[0]), str(panel.panel_rect[1]), str(panel.panel_rect[2]), str(panel.panel_rect[3]),
                                    'panel'])


if __name__ == '__main__':
    main()
