import torchvision.transforms.v2.functional as tvf

STANDARD_SIZE = 128

def standardize(img_tsr):
    """Crop and resize images to 128x128."""
    less_dim = min(img_tsr.shape[-2:])
    img_tsr = tvf.center_crop(img_tsr, less_dim)
    img_tsr = tvf.resize(img_tsr, (STANDARD_SIZE, STANDARD_SIZE))
    return img_tsr.float()/255
