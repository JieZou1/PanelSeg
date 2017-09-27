
def make_square(rect):
    """
    Extend rectangle to make a square
    :param rect:
    :return:
    """
    [x, y, w, h] = rect
    if w == h:
        return rect
    elif w > h:
        h_new = w
        c = y + h / 2
        y = c - h_new / 2
        h = h_new
    else:
        w_new = h
        c = x + w / 2
        x = c - w_new / 2
        w = w_new

    return [round(x), round(y), w, h]


def extend_square(square, upper_percent, lower_percent):
    s = square[2]
    s_upper = s * (1+upper_percent)
    s_lower = s * (1-lower_percent)

    c_x = square[0] + s / 2
    c_y = square[1] + s / 2

    square_upper = [round(c_x-s_upper/2), round(c_y-s_upper/2), round(s_upper), round(s_upper)]
    square_lower = [round(c_x-s_lower/2), round(c_y-s_lower/2), round(s_lower), round(s_lower)]
    return square_upper, square_lower


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
