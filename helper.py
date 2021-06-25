import numpy as np


def get_segments_from_row(row):
    borders = row[1:] != row[:-1]
    borders = np.append(borders, True)
    indices = np.where(borders)[0] + 1
    segments = np.split(row, indices)
#     segments = np.array([elem.flatten() for elem in segments], dtype=object)[:-1]
    segments = list(elem.flatten() for elem in segments)[:-1]
    segments = segments[1:-1] # rm lines on edges
    return segments


def get_segments_lengths_from_image(img):
    result = {}
    for d in np.arange(img.ndim):
        true_lengths = np.array([])
        false_lengths = np.array([])
        stripes = np.split(img, img.shape[d], axis=d)
        for stripe in stripes:
            segments = get_segments_from_row(stripe.ravel())
            true_segments = filter(lambda x: True in x, segments)
            false_segments = filter(lambda x: False in x, segments)
            for ts in true_segments:
                true_lengths = np.append(true_lengths, len(ts))
            for fs in false_segments:
                false_lengths = np.append(false_lengths, len(fs))
        result[d] = {'pores': false_lengths, 'solid': true_lengths}        
    print(result)
