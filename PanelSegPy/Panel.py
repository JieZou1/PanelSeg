
class Panel:
    """
    A class for Panel
    """

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

    def __init__(self, label, panel_rect, label_rect):
        self.label = label
        self.panel_rect = panel_rect    #list [x, y, w, h]
        self.label_rect = label_rect    #list [x, y, w, h]
        self.panel_patch = None
        self.label_patch = None

        self.label_patches = None   # used for offline training for holding small variations from gt label annotations
        self.label_rects = None

