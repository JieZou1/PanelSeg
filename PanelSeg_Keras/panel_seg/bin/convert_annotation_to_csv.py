import argparse
import sys
import os
import csv
from figure.figure_set import FigureSet
from figure.figure import Figure
from figure.misc import map_label


def parse_args(args):
    parser = argparse.ArgumentParser(description='Convert PanelSeg data set to CSV format.')
    parser.add_argument('--list_path',
                        help='The path to the list file.',
                        default='\\Users\\jie\\projects\\PanelSeg\\ExpKeras\\all.txt',
                        type=str)
    parser.add_argument('--annotation_path',
                        help='The output annotation CSV file.',
                        default='\\Users\\jie\\projects\\PanelSeg\\ExpKeras\\panel_seg\\all.csv',
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
            # The format is:
            # image_path,panel_x1,panel_y1,panel_x2,panel_y2,label_x1,label_y1,label_x2,label_y2,label
            # if there is no label, the format becomes:
            # image_path,panel_x1,panel_y1,panel_x2,panel_y2,,,,,
            for panel in figure.panels:
                row = list()
                row.append(figure.image_path)  # add image_path
                row.append(str(panel.panel_rect[0]))
                row.append(str(panel.panel_rect[1]))
                row.append(str(panel.panel_rect[2]))
                row.append(str(panel.panel_rect[3]))
                row.append('panel')
                if panel.label_rect is None:
                    row.append('')
                    row.append('')
                    row.append('')
                    row.append('')
                    row.append('')
                else:
                    label = map_label(panel.label)
                    row.append(str(panel.label_rect[0]))
                    row.append(str(panel.label_rect[1]))
                    row.append(str(panel.label_rect[2]))
                    row.append(str(panel.label_rect[3]))
                    row.append(label)
                csv_writer.writerow(row)


if __name__ == '__main__':
    main()
