import numpy as np
from scipy import stats
from scipy import ndimage
import pandas as pd

import cv2

import matplotlib.pyplot as plt
from matplotlib import colors


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


def image_statistics(img, remove_edges=False):
    result = { 'segments_lengths': {}, 'edge_distances': {} }
    edge_distances_result = {}
    for d in np.arange(img.ndim):
        
        solid_lengths = np.array([], dtype=np.int32)
        pores_lengths = np.array([], dtype=np.int32)
        
        edge_distances_result[d] = np.ma.masked_all(img.shape, dtype=np.int32)
        
        stripes = np.split(img, img.shape[d], axis=d)
        for index, stripe in enumerate(stripes):
            
            segments = segments_from_row(stripe.ravel(), remove_edges=remove_edges)
            
            edge_distances = [[idx for idx, _ in enumerate(segment)] for segment in segments]
            # https://stackoverflow.com/questions/11264684/flatten-list-of-lists
            edge_distances = np.ma.array([val for sublist in edge_distances for val in sublist])
            first_segment_length = segments[0].size
            last_segment_length = segments[-1].size
            edge_distances[:first_segment_length] = np.ma.masked
            edge_distances[-last_segment_length:] = np.ma.masked
            edge_distances_result[d][index] = edge_distances
            
            if remove_edges:
                segments = segments[1:-1]
            true_segments = filter(lambda x: True in x, segments)
            false_segments = filter(lambda x: False in x, segments)
            for ts in true_segments:
                solid_lengths = np.append(solid_lengths, len(ts))
            for fs in false_segments:
                pores_lengths = np.append(pores_lengths, len(fs))
                
        result['segments_lengths'][d] = {'pores': np.sort(pores_lengths), 'solid': np.sort(solid_lengths)}
    result['edge_distances'] = edge_distances_result
    return result


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


def edge_distances_from_image(img):
    result = {}
    for d in np.arange(img.ndim):
        stripes = np.split(img, img.shape[d], axis=d)
        result[d] = np.ma.zeros(img.shape)
        for index, stripe in enumerate(stripes):
            segments = segments_from_row(stripe.ravel())
            edge_distances = [[idx for idx, _ in enumerate(segment)] for segment in segments]
            # https://stackoverflow.com/questions/11264684/flatten-list-of-lists
            edge_distances = np.ma.array([val for sublist in edge_distances for val in sublist])
            first_segment_length = segments[0].size
            last_segment_length = segments[-1].size
            edge_distances[:first_segment_length] = np.ma.masked
            edge_distances[-last_segment_length:] = np.ma.masked
            result[d][index] = edge_distances
    return result


def hist_of_lengths(segments_lengths, density=True):
    max_value = np.max(segments_lengths) + 1
    hist, edges = np.histogram(segments_lengths, range=(0, max_value), bins=max_value, density=density)
#     print(f'hist: { hist }, hist sum: { np.sum(hist) }')
#     print(f'edges: { edges }')
    cdf_values = [np.sum(hist[:i + 1]) for i in np.arange(0, max_value + 1)]
    def hist_cdf(x):
        x = 0 if x < 0 else x
        x = max_value if x > max_value else x
        return cdf_values[np.int32(x)]
    return hist, edges, hist_cdf


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

    
def calc_2d_hists(solid_data, void_data, show=True, drop_zero=False):

    s_y = solid_data[:, 0]
    s_x = solid_data[:, 1]
    max_s_y = np.max(s_y) + 1
    max_s_x = np.max(s_x) + 1
    s_hist, _, _ = np.histogram2d(s_y, s_x, bins=(max_s_y, max_s_x), range=[[0, max_s_y], [0, max_s_x]])
    s_hist = np.int32(s_hist)
    if drop_zero:
        s_hist[0, 0] = 0
    # print(s_hist)
    
    v_y = void_data[:, 0]
    v_x = void_data[:, 1]
    max_v_y = np.max(v_y) + 1
    max_v_x = np.max(v_x) + 1
    v_hist, _, _ = np.histogram2d(v_y, v_x, bins=(max_v_y, max_v_x), range=[[0, max_v_y], [0, max_v_x]])
    v_hist = np.int32(v_hist)
    if drop_zero:
        v_hist[0, 0] = 0
    # print(v_hist)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        im0 = axes[0].imshow(s_hist, norm=colors.LogNorm())
        axes[1].imshow(v_hist, norm=colors.LogNorm())
        middle_s = (np.max(s_hist) - 1) / 2
        middle_v = (np.max(v_hist) - 1) / 2

        if np.max((*s_hist.shape, *v_hist.shape)) < 20:
        
            for i in range(s_hist.shape[0]):
                for j in range(s_hist.shape[1]):
                    color = 'black' if s_hist[i, j] > middle_s else 'w'
                    axes[0].text(j, i, s_hist[i, j], ha='center', va='center', c=color, size='xx-large')

            for i in range(v_hist.shape[0]):
                for j in range(v_hist.shape[1]):
                    color = 'black' if v_hist[i, j] > middle_v else 'w'
                    axes[1].text(j, i, v_hist[i, j], ha='center', va='center', c=color, size='xx-large')
        
    return s_hist, v_hist


def dataframes_from_image_with_nan_at_edges(img):
    
    im_stats = image_statistics(img)
    edge_distances = im_stats['edge_distances']
    eds_y = (np.int32(edge_distances[1]) + 1).T
    eds_x = np.int32(edge_distances[0]) + 1

    y_size = img.shape[-2]
    x_size = img.shape[-1]
    y_grid = np.arange(y_size)
    x_grid = np.arange(x_size)
    indices = pd.MultiIndex.from_tuples(list(np.ndindex(y_size, x_size)))

    df = pd.DataFrame(columns=['isSolid', 'leftLength', 'leftIsSolid', 'topLength', 'topIsSolid'], index=indices)

    for y in y_grid:
        for x in x_grid:

            is_solid = bool(img[y, x])
            left_length = np.NaN
            left_is_solid = np.NaN
            rigth_length = np.NaN
            rigth_is_solid = np.NaN

            if x > 0:
                prev_x = x - 1
                is_masked = eds_x.mask[y, prev_x]
                if not is_masked:
                    left_length = eds_x.data[y, prev_x]
                    left_is_solid = bool(img[y, prev_x])

            if y > 0:
                prev_y = y - 1
                is_masked = eds_y.mask[prev_y, x]
                if not is_masked:
                    rigth_length = eds_y.data[prev_y, x]
                    rigth_is_solid = bool(img[prev_y, x])

            df.loc[(y, x)] = pd.Series({
                'isSolid': is_solid, 
                'leftLength': left_length, 
                'leftIsSolid': left_is_solid, 
                'topLength': rigth_length, 
                'topIsSolid': rigth_is_solid,
            })

    print(df.info())
    print(df.shape)

    dff = df[df.notna().all(axis='columns')].astype(np.int32)

    print(dff.info())
    print(dff.shape)
    
    return df, dff


def dataframe_from_image(img):
    
    im_stats = image_statistics(img)
    edge_distances = im_stats['edge_distances']
    eds_y = np.int32(edge_distances[1]).T
    eds_y = eds_y.data + 1
    eds_x = np.int32(edge_distances[0])
    eds_x = eds_x.data + 1

    y_size = img.shape[-2]
    x_size = img.shape[-1]
    y_grid = np.arange(y_size)
    x_grid = np.arange(x_size)
    indices = pd.MultiIndex.from_tuples(list(np.ndindex(y_size, x_size)))

    df = pd.DataFrame(columns=['isSolid', 'leftLength', 'leftIsSolid', 'topLength', 'topIsSolid'], index=indices)

    for y in y_grid:
        for x in x_grid:

            is_solid = bool(img[y, x])
            left_length = 0
            left_is_solid = -1
            rigth_length = 0
            rigth_is_solid = -1

            if x > 0:
                prev_x = x - 1
                left_length = eds_x.data[y, prev_x]
                left_is_solid = bool(img[y, prev_x])

            if y > 0:
                prev_y = y - 1
                rigth_length = eds_y.data[prev_y, x]
                rigth_is_solid = bool(img[prev_y, x])

            df.loc[(y, x)] = pd.Series({
                'isSolid': is_solid, 
                'leftLength': left_length, 
                'leftIsSolid': left_is_solid, 
                'topLength': rigth_length, 
                'topIsSolid': rigth_is_solid,
            })

    df = df.astype(np.int32)

    print(df.info())
    print(df.shape)
    
    return df


def pattern_dataframe_from_image(img):
    
    img = img.astype(np.int32)
    y_size = img.shape[-2]
    x_size = img.shape[-1]
    y_grid = np.arange(y_size)
    x_grid = np.arange(x_size)
    indices = pd.MultiIndex.from_tuples(list(np.ndindex(y_size, x_size)))

    df = pd.DataFrame(columns=['isSolid', 'left', 'topleft', 'top', 'topright', 'pattern'], index=indices)

    for y in y_grid:
        for x in x_grid:

            is_solid = img[y, x]

            left = -1
            topleft = -1
            top = -1
            topright = -1
                            
            prev_x = x - 1
            next_x = x + 1

            prev_y = y - 1

            if y == 0 and x == 0:
                pass
            elif y == 0:
                left = img[y, prev_x]
            elif x == 0:
                top = img[prev_y, x]
                topright = img[prev_y, next_x]
            elif x == x_size - 1:
                left = img[y, prev_x]
                topleft = img[prev_y, prev_x]
                top = img[prev_y, x]
            else:
                left = img[y, prev_x]
                topleft = img[prev_y, prev_x]
                top = img[prev_y, x]
                topright = img[prev_y, next_x]
                
            df.loc[(y, x)] = pd.Series({
                'isSolid': is_solid,
                'left': left, 
                'topleft': topleft, 
                'top': top, 
                'topright': topright,
                'pattern': ''.join([str(i) for i in [left, topleft, top, topright]]),
            })

    df[['isSolid', 'left', 'topleft', 'top', 'topright']] = df[['isSolid', 'left', 'topleft', 'top', 'topright']].astype(np.int32)

    print(df.info())
    print(df.shape)
    
    return df


def borders_from_image(img):
    x_borders = None
    x_borders_solid = None
    x_borders_void = None
    y_borders = None
    y_borders_solid = None
    y_borders_void = None

    for y in np.arange(img.shape[-2]):
        row = img[y, :]
        borders = row[1:] != row[:-1]
        borders = np.append(True, borders)
        indices = np.where(borders)[0]
        points = [(y, idx) for idx in indices]
        x_borders = points if x_borders is None else np.append(x_borders, points, axis=0)
        x_borders_solid = np.array([(y, x) for y, x in x_borders if img[y, x] == 1])
        x_borders_void = np.array([(y, x) for y, x in x_borders if img[y, x] == 0])

    for x in np.arange(img.shape[-1]):
        row = img[:, x]
        borders = row[1:] != row[:-1]
        borders = np.append(True, borders)
        indices = np.where(borders)[0]
        points = [(idy, x) for idy in indices]
        y_borders = points if y_borders is None else np.append(y_borders, points, axis=0)
        y_borders_solid = np.array([(y, x) for y, x in y_borders if img[y, x] == 1])
        y_borders_void = np.array([(y, x) for y, x in y_borders if img[y, x] == 0])
        
    return x_borders_solid, x_borders_void, y_borders_solid, y_borders_void


def border_distances_for_image(img):
    
    x_distances_solid = np.empty(img.shape, dtype=np.int32)
    x_distances_void = np.empty(img.shape, dtype=np.int32)
    y_distances_solid = np.empty(img.shape, dtype=np.int32)
    y_distances_void = np.empty(img.shape, dtype=np.int32)

    x_borders_solid, x_borders_void, y_borders_solid, y_borders_void = borders_from_image(img)
    
    def get_distance(point, borders, direction='x'):
        fixed = -2 if direction == 'x' else -1
        searched = -1 if direction == 'x' else -2
        distances = [
            point[searched] - b[searched] for b in borders if b[fixed] == point[fixed] and b[searched] <= point[searched]
        ]
        return np.min(distances) if len(distances) else -1

    for y in np.arange(img.shape[-2]):
        for x in np.arange(img.shape[-1]):
            p = (y, x)
            x_distances_solid[y, x] = get_distance(p, x_borders_solid)
            x_distances_void[y, x] = get_distance(p, x_borders_void)
            y_distances_solid[y, x] = get_distance(p, y_borders_solid, direction='y')
            y_distances_void[y, x] = get_distance(p, y_borders_void, direction='y')

    return x_distances_solid, x_distances_void, y_distances_solid, y_distances_void
