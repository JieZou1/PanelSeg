
class PanelSegError(Exception):
    """
    Exception for FigureSeg
    """
    def __init__(self, message):
        self.message = message
