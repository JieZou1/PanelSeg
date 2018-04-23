"""Core Classes and Functions for PanelSeg."""


LABEL_CLASS_MAPPING = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    'A': 9,
    'B': 10,
    'C': 11,
    'D': 12,
    'E': 13,
    'F': 14,
    'G': 15,
    'H': 16,
    'I': 17,
    'J': 18,
    'K': 19,
    'L': 20,
    'M': 21,
    'N': 22,
    'O': 23,
    'P': 24,
    'Q': 25,
    'R': 26,
    'S': 27,
    'T': 28,
    'U': 29,
    'V': 30,
    'W': 31,
    'X': 32,
    'Y': 33,
    'Z': 34,
    'a': 35,
    'b': 36,
    'd': 37,
    'e': 38,
    'f': 39,
    'g': 40,
    'h': 41,
    'i': 42,
    'j': 43,
    'l': 44,
    'm': 45,
    'n': 46,
    'q': 47,
    'r': 48,
    't': 49,
}
CLASS_LABEL_MAPPING = {v: k for k, v in LABEL_CLASS_MAPPING.items()}


class Panel:
    """
    A class for a Panel
    """
    def __init__(self):
        pass
    pass


class Figure:
    """
    A class for a Figure
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    panels contain all panels
    """
    def __init__(self):
        pass

    pass


class FigureSet:
    """
    A class for a FigureSet
    image_path is the path to the figure image file
    id is the unique id to each figure
    image_orig is the original color image
    panels contain all panels
    """

    def __init__(self, list_file):
        self.list_file = list_file

    pass

