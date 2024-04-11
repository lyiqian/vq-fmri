import torch

from models import (
    TokenNoising,
    ImageEncoder,
    ImageDecoder,
    VectorQuantizer,
    TokenClassifier,
    InpaintingNetwork,
    VqVae,
    VqVaeAbc,
    FMRIEncoder,
    UNet,
    SuperResolutionModule,
    MLP,
)
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from data import GODLoader, ImageLoader
from typing import Type
import itertools

from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as TF

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device in use is {}".format(device))
device = torch.device(device)

def inference(
    godloader: GODLoader
):
    fmri_enc = FMRIEncoder(in_dims=4466, out_width=16)
    vq_vae_s = VqVae()
    vq_vae_l = VqVae(codebook_size=256)
    vq_vae_fmri = VqVae()
    token_classifier = TokenClassifier()
    inpainting_network = InpaintingNetwork(dim_encodings=8)
    token_classifier.load_state_dict(torch.load('/home/brouhani/Documents/courses/dl/vq-fmri/src/states/phase1/apr6-full-vqvae/phase3/token_classifier_356000.pth'))
    inpainting_network.load_state_dict(torch.load('/home/brouhani/Documents/courses/dl/vq-fmri/src/states/phase1/apr6-full-vqvae/phase3/token_inpainting_356000.pth'))
    sr_module = SuperResolutionModule(vq_vae_s, vq_vae_l, vq_vae_fmri, token_classifier, inpainting_network, img_size=[256, 256])
    sr_module.token_classifier = token_classifier
    sr_module.inpainting_network = inpainting_network
    fmri_enc.load_state_dict(torch.load('./states/phase2/fmri-encoder-epoch-389.pth'))
    sd = torch.load('./states/phase1/apr6-full-vqvae/sr/sr_module_74000.pth')
    # print(sd.keys())
    # exit()
    sr_module.load_state_dict(sd, strict=False)
    sr_module.to(device)
    fmri_enc.to(device)
    for batch_idx, (image, fmri) in tqdm(enumerate(godloader)):
        fmri = fmri.to(device)
        image = image.to(device)
        # load models
        # produce results
        encoded_fmri = fmri_enc.encode(fmri)
        output = sr_module(encoded_fmri)
        combined = torch.empty((output.size(0) * 2, output.size(1), output.size(2), output.size(3)), dtype=image.dtype)
        # left = right = top = bottom = (512 - 128) // 2
        # Pad the image
        # padded_image = TF.pad(image, padding=(left, top, right, bottom), padding_mode='constant', fill=0)
        padded_image = TF.resize(image, size=(128, 128))
        combined[0::2] = padded_image.detach().cpu()
        combined[1::2] = output.detach().cpu()

        grid = make_grid(combined, nrow=2)

        # Save the grid of images
        save_image(grid, f'./saved_images/batch_{batch_idx}_grid.png')        
            # log results and metrics
