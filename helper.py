import numpy as np
from scipy import stats


def segments_from_row(row):
    borders = row[1:] != row[:-1]
    borders = np.append(borders, True)
    indices = np.where(borders)[0] + 1
    segments = np.split(row, indices)
#     segments = np.array([elem.flatten() for elem in segments], dtype=object)[:-1]
    segments = list(elem.flatten() for elem in segments)[:-1]
    segments = segments[1:-1] # rm lines on edges
    return segments


def segments_lengths_from_image(img):
    result = {}
    for d in np.arange(img.ndim):
        true_lengths = np.array([], dtype=np.int32)
        false_lengths = np.array([], dtype=np.int32)
        stripes = np.split(img, img.shape[d], axis=d)
        for stripe in stripes:
            segments = segments_from_row(stripe.ravel())
            true_segments = filter(lambda x: True in x, segments)
            false_segments = filter(lambda x: False in x, segments)
            for ts in true_segments:
                true_lengths = np.append(true_lengths, len(ts))
            for fs in false_segments:
                false_lengths = np.append(false_lengths, len(fs))
        result[d] = {'pores': false_lengths, 'solid': true_lengths}        
    return result


def hist_of_lengths(segments_lengths):
    max_value = np.max(segments_lengths)
    hist, edges = np.histogram(segments_lengths, bins=max_value, density=True)
    return hist, edges


def kde_of_lengths(segments_lengths):
    max_value = np.max(segments_lengths)
    kde = stats.gaussian_kde(segments_lengths)
    linspace = np.linspace(1, max_value, num=max_value, dtype=np.int32)
    pdf = kde.pdf(linspace)
    cdf_values = [np.sum(pdf[:i]) for i in linspace]
    def cdf(x):
        return 0 if x <= 0 else None if x >= max_value else cdf_values[np.int32(x)]
    return kde, pdf, cdf, linspace


def get_sample(kde):
    result = np.int32(np.ceil(kde.resample(1)))[0]
    return result if result > 0 else get_sample(kde) #TODO: recursion!!! need to break infinite loop if have one
