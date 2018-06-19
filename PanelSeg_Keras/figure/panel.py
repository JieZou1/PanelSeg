
class Panel:
    """
    A class for a Panel
    """
    def __init__(self, label, panel_rect, label_rect):
        self.label = label
        self.panel_rect = panel_rect    # list [x_min, y_min, x_max, y_max]
        self.label_rect = label_rect    # list [x_min, y_min, x_max, y_max]
