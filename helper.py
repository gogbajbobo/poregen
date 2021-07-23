import numpy as np
from scipy import stats
from scipy import ndimage
import cv2


def segments_from_row(row, remove_edges=False):
    borders = row[1:] != row[:-1]
    borders = np.append(borders, True)
    indices = np.where(borders)[0] + 1
    segments = np.split(row, indices)
#     segments = np.array([elem.flatten() for elem in segments], dtype=object)[:-1]
    segments = list(elem.flatten() for elem in segments)[:-1]
    if remove_edges:
        segments = segments[1:-1]
    return segments


def segments_lengths_from_image(img):
    result = {}
    for d in np.arange(img.ndim):
        true_lengths = np.array([], dtype=np.int32)
        false_lengths = np.array([], dtype=np.int32)
        stripes = np.split(img, img.shape[d], axis=d)
        for stripe in stripes:
            segments = segments_from_row(stripe.ravel(), remove_edges=True)
            true_segments = filter(lambda x: True in x, segments)
            false_segments = filter(lambda x: False in x, segments)
            for ts in true_segments:
                true_lengths = np.append(true_lengths, len(ts))
            for fs in false_segments:
                false_lengths = np.append(false_lengths, len(fs))
        result[d] = {'pores': np.sort(false_lengths), 'solid': np.sort(true_lengths)}        
    return result


def hist_of_lengths(segments_lengths):
    max_value = np.max(segments_lengths) + 1
    hist, edges = np.histogram(segments_lengths, range=(0, max_value), bins=max_value, density=True)
#     print(f'hist: { hist }, hist sum: { np.sum(hist) }')
#     print(f'edges: { edges }')
    return hist, edges


def kde_of_lengths(segments_lengths):
    max_value = np.max(segments_lengths) +1
    kde = stats.gaussian_kde(segments_lengths)
    linspace = np.linspace(0, max_value, num=max_value + 1, dtype=np.int32)
#     print(f'linspace: { linspace }')
    pdf = kde.pdf(linspace)
    pdf[0] = 0
    pdf[max_value] = 0
    pdf = pdf / np.sum(pdf)
#     print(f'pdf: { pdf }')
    cdf_values = [np.sum(pdf[:i + 1]) for i in linspace]
#     print(f'cdf_values: { cdf_values }')
    def cdf(x):
        x = 0 if x < 0 else x
        x = max_value if x > max_value else x
        return cdf_values[np.int32(x)]
    return kde, pdf, cdf, linspace


def get_sample(kde):
    result = np.int32(np.ceil(kde.resample(1)))[0]
    return result if result > 0 else get_sample(kde) #TODO: recursion!!! need to break infinite loop if have one


def image_porosity(img):
    return 1 - np.sum(img) / img.size


def one_D_generator(size=128, sigma=4, porosity=0.5, seed=None):
    np.random.seed(seed)
    image = np.random.random((size)).astype(np.float32)
    gauss_image = ndimage.gaussian_filter(image, sigma=sigma, truncate=4)
    img = cv2.normalize(gauss_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.equalizeHist(img) / 255
    return img > porosity


def compare_lengths(h1, h2):
    print('Modes:')
    print(stats.mode(h1['pores']))
    print(stats.mode(h1['solid']))
    print(stats.mode(h2['pores']))
    print(stats.mode(h2['solid']))
    print('\n')
    print('Means:')
    print(np.mean(h1['pores']))
    print(np.mean(h1['solid']))
    print(np.mean(h2['pores']))
    print(np.mean(h2['solid']))
    print('\n')
    print('STDs:')
    print(np.std(h1['pores']))
    print(np.std(h1['solid']))
    print(np.std(h2['pores']))
    print(np.std(h2['solid']))
    print('\n')
    print('Medians:')
    print(np.median(h1['pores']))
    print(np.median(h1['solid']))
    print(np.median(h2['pores']))
    print(np.median(h2['solid']))
    print('\n')
            

def compare_hists(h1, h2):
    print('Sum diffs:')
    p_hists_diff = np.zeros(np.max([h1['pores'].size, h2['pores'].size]))
    p_hists_diff[:h1['pores'].size] = h1['pores']
    p_hists_diff[:h2['pores'].size] -= h2['pores']

    s_hists_diff = np.zeros(np.max([h1['solid'].size, h2['solid'].size]))
    s_hists_diff[:h1['solid'].size] = h1['solid']
    s_hists_diff[:h2['solid'].size] -= h2['solid']
    
    print(np.sum(p_hists_diff))
    print(np.sum(s_hists_diff))
    print('\n')
    
    print('CorrCoefs:')
    p_eq_hist = np.zeros(np.max([h1['pores'].size, h2['pores'].size]))
    t_p_eq_hist = np.zeros(np.max([h1['pores'].size, h2['pores'].size]))
    p_eq_hist[:h1['pores'].size] = h1['pores']
    t_p_eq_hist[:h2['pores'].size] = h2['pores']

    s_eq_hist = np.zeros(np.max([h1['solid'].size, h2['solid'].size]))
    t_s_eq_hist = np.zeros(np.max([h1['solid'].size, h2['solid'].size]))
    s_eq_hist[:h1['solid'].size] = h1['solid']
    t_s_eq_hist[:h2['solid'].size] = h2['solid']

    print(np.corrcoef(p_eq_hist, t_p_eq_hist))
    print(np.corrcoef(s_eq_hist, t_s_eq_hist))
    print('\n')
