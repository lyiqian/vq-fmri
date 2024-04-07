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


arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 2, 4, 4])
pcc(arr1, arr2)
