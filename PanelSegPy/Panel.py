
class Panel:
    """
    A class for Panel
    """

    def __init__(self, label, panel_rect, label_rect):
        self.label = label
        self.panel_rect = panel_rect    #list [x, y, w, h]
        self.label_rect = label_rect    #list [x, y, w, h]
        self.panel_patch = None
        self.label_patch = None

        self.label_patches = None   # used for offline training for holding small variations from gt label annotations
        self.label_rects = None

