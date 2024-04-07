import numpy as np


def pcc(orig: np.ndarray, rec: np.ndarray):
    """pixel-wise Pearson correlation coefficients."""
    orig = orig.ravel()
    rec = rec.ravel()

    orig_centered = orig - orig.mean()
    rec_centered = rec - rec.mean()

    pcc = (
        (orig_centered * rec_centered).sum()
        / np.sqrt((orig_centered**2).sum() * (rec_centered**2).sum())
    )
    return pcc


SSIM_WINDOW_SIZE = 3


def ssim(orig: np.ndarray, rec: np.ndarray):
    """Structural similarity index.

    Neither the main paper nor this one mentioned 'window size', so just using
    3x3 for now: https://arxiv.org/pdf/2110.09006.pdf
    """
    eps1, eps2 = 1e-6, 1e-6
    rad = SSIM_WINDOW_SIZE // 2

    print(orig.shape)
    local_ssim = []
    for i in range(rad, orig.shape[0]-rad):
        for j in range(rad, orig.shape[1]-rad):
            for c in range(3):
                orig_l = orig[i-rad:i+1+rad, j-rad:j+1+rad, c].ravel()
                rec_l = rec[i-rad:i+1+rad, j-rad:j+1+rad, c].ravel()

                orig_mean, rec_mean = orig_l.mean(), rec_l.mean()
                orig_var, rec_var = orig_l.var(), rec_l.var()
                covar = np.cov(orig_l, rec_l)[0][1]

                result = (
                    (2*orig_mean*rec_mean + eps1) * (2*covar + eps2)
                    / ((orig_mean**2 + rec_mean**2 + eps1) * (orig_var + rec_var + eps2))
                )
                local_ssim.append(result)

    global_ssim = np.mean(local_ssim)
    return global_ssim


# import pathlib
# import matplotlib.pyplot as plt

# DATA_DIR = pathlib.Path(__file__).parent.parent / 'data'

# img1 = plt.imread(str(DATA_DIR/'rec1.png'))
# img2 = plt.imread(str(DATA_DIR/'rec2.png'))

# print("PCC", pcc(img1, img2))
# print("SSIM", ssim(img1, img2))
