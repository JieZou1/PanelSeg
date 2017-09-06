from Figure import Figure
from Panel import Panel


def generate_annotation_preview(list_file="/Users/jie/projects/PanelSeg/Exp/all.txt"):
    with open(list_file) as f:
        files = f.readlines()
        # Remove whitespace characters, and then construct the annotation filename
        files = [x.strip() for x in files]

    for file in files:
        figure = Figure(file)
        figure.load_gt_annotation_iphotodraw()

if __name__ == "__main__":
    generate_annotation_preview()
