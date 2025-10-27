from scipy.ndimage import gaussian_filter1d

def smooth_holiday(hol_ind):
    smooth_h = gaussian_filter1d(hol_ind, sigma=0.65)
    return smooth_h / smooth_h.max()