
LABEL_MIN_SIZE = 12
LABEL_MAX_SIZE = 80
LABEL_ALL = '123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'

# Map labels to the folder where the sample patches are stored.
LABEL_FOLDER_MAPPING = {
    '1': ['[49]'],
    '2': ['[50]'],
    '3': ['[51]'],
    '4': ['[52]'],
    '5': ['[53]'],
    '6': ['[54]'],
    '7': ['[55]'],
    '8': ['[56]'],
    '9': ['[57]'],
    'A': ['[65]'],
    'B': ['[66]'],
    'C': ['[67]', '[99]'],
    'D': ['[68]'],
    'E': ['[69]'],
    'F': ['[70]'],
    'G': ['[71]'],
    'H': ['[72]'],
    'I': ['[73]'],
    'J': ['[74]'],
    'K': ['[75]', '[107]'],
    'L': ['[76]'],
    'M': ['[77]'],
    'N': ['[78]'],
    'O': ['[79]', '[111]'],
    'P': ['[80]', '[112]'],
    'Q': ['[81]'],
    'R': ['[82]'],
    'S': ['[83]', '[115]'],
    'T': ['[84]'],
    'U': ['[85]'],
    'V': ['[86]', '[118]'],
    'W': ['[87]', '[119]'],
    'X': ['[88]', '[120]'],
    'Y': ['[89]'],
    'Z': ['[122]'],
    'a': ['[97]'],
    'b': ['[98]'],
    'd': ['[100]'],
    'e': ['[101]'],
    'f': ['[102]'],
    'g': ['[103]'],
    'h': ['[104]'],
    'i': ['[105]'],
    'j': ['[106]'],
    'l': ['[108]'],
    'm': ['[109]'],
    'n': ['[110]'],
    'q': ['[113]'],
    'r': ['[114]'],
    't': ['[116]'],
}

LABEL_MAPPING = {
    'fg': 0,
}

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


def case_same_label(c):
    if c == 'c' or c == 'C':
        return True
    elif c == 'k' or c == 'K':
        return True
    elif c == 'o' or c == 'O':
        return True
    elif c == 'p' or c == 'P':
        return True
    elif c == 's' or c == 'S':
        return True
    elif c == 'u' or c == 'U':
        return True
    elif c == 'v' or c == 'V':
        return True
    elif c == 'w' or c == 'W':
        return True
    elif c == 'x' or c == 'X':
        return True
    elif c == 'y' or c== 'Y':
        return True
    elif c == 'z' or c == 'Z':
        return True
    else:
        return False


def map_label(c):
    if c == 'c' or c == 'C':
        return 'C'
    elif c == 'k' or c == 'K':
        return 'K'
    elif c == 'o' or c == 'O':
        return 'O'
    elif c == 'p' or c == 'P':
        return 'P'
    elif c == 's' or c == 'S':
        return 'S'
    elif c == 'u' or c == 'U':
        return 'U'
    elif c == 'v' or c == 'V':
        return 'V'
    elif c == 'w' or c == 'W':
        return 'W'
    elif c == 'x' or c == 'X':
        return 'X'
    elif c == 'y' or c== 'Y':
        return 'Y'
    elif c == 'z' or c == 'Z':
        return 'Z'
    else:
        return c


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

