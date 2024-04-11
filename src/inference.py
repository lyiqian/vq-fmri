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

from evaluation import ssim, psnr, pcc

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Device in use is {}".format(device))
device = torch.device(device)

def inference(
    godloader: GODLoader,
    models_dir:str
):
    models_dir = Path(models_dir)

    fmri_enc = FMRIEncoder(in_dims=4466, out_width=16)
    vq_vae_s = VqVae()
    vq_vae_l = VqVae(codebook_size=256)
    vq_vae_fmri = VqVae()
    token_classifier = TokenClassifier()
    inpainting_network = InpaintingNetwork(dim_encodings=8)
    sr_module = SuperResolutionModule(vq_vae_s, vq_vae_l, vq_vae_fmri, token_classifier, inpainting_network, img_size=[256, 256])

    token_classifier.load_state_dict(torch.load(models_dir / 'token_classifier_128000.pth'))
    inpainting_network.load_state_dict(torch.load(models_dir / 'token_inpainting_128000.pth'))
    fmri_enc.load_state_dict(torch.load(models_dir / 'fmri-encoder-epoch-389.pth'))
    sr_module.load_state_dict(torch.load(models_dir / 'sr_module_8000.pth'), strict=False)
    sr_module.token_classifier = token_classifier
    sr_module.inpainting_network = inpainting_network

    token_classifier.eval()
    inpainting_network.eval()
    sr_module.eval()
    fmri_enc.eval()
    sr_module.to(device)
    fmri_enc.to(device)

    results = []
    targets = []
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
        save_image(grid, f'./saved_images/test_batch_{batch_idx}_grid.png')        
            # log results and metrics
        results += [output[i].detach().cpu().numpy() for i in range(output.shape[0])]
        targets += [padded_image[i].detach().cpu().numpy() for i in range(output.shape[0])]
        if batch_idx > 4:
            break


    psnr_all = 0
    pcc_all = 0
    ssim_all = 0
    len_data = len(results)
    for result, target in zip(results, targets):
        sample_pcc = pcc(target, result)
        pcc_all += sample_pcc
        sample_psnr = psnr(target, result)
        psnr_all += sample_psnr
        sample_ssim = ssim(target, result) 
        ssim_all += sample_ssim

    print('#########################')    
    print('##Final metric results:##')
    print('#########################')    
    print('PSNR MEAN: {}'.format(psnr_all/len_data))
    print('SSIM MEAN: {}'.format(ssim_all/len_data))
    print('PCC MEAN: {}'.format(pcc_all/len_data))



def inference_no_sr(
    godloader: GODLoader,
    models_dir:str
):
    models_dir = Path(models_dir)

    fmri_enc = FMRIEncoder(in_dims=4466, out_width=16)
    vq_vae_s = VqVae()
    vq_vae_l = VqVae(codebook_size=256)
    vq_vae_fmri = VqVae()
    token_classifier = TokenClassifier()
    inpainting_network = InpaintingNetwork(dim_encodings=8)
    sr_module = SuperResolutionModule(vq_vae_s, vq_vae_l, vq_vae_fmri, token_classifier, inpainting_network, img_size=[256, 256])

    token_classifier.load_state_dict(torch.load(models_dir / 'token_classifier_128000.pth'))
    inpainting_network.load_state_dict(torch.load(models_dir / 'token_inpainting_128000.pth'))
    fmri_enc.load_state_dict(torch.load(models_dir / 'fmri-encoder-epoch-389.pth'))
    sr_module.load_state_dict(torch.load(models_dir / 'sr_module_8000.pth'), strict=False)
    sr_module.token_classifier = token_classifier
    sr_module.inpainting_network = inpainting_network

    # for inference with smaller vq-vae
    # vq_vae = VqVae(codebook_size=32)
    # vq_vae.load(models_dir / 'small', 4)
    vq_vae = VqVae()
    vq_vae.load(models_dir, 4)
    vq_vae.eval()
    vq_vae.to(device)

    token_classifier.eval()
    inpainting_network.eval()
    sr_module.eval()
    fmri_enc.eval()
    sr_module.to(device)
    fmri_enc.to(device)

    results = []
    targets = []
    for batch_idx, (image, fmri) in tqdm(enumerate(godloader)):
        fmri = fmri.to(device)
        image = image.to(device)
        # load models
        # produce results
        encoded_fmri = fmri_enc.encode(fmri)
        output = vq_vae.decode(encoded_fmri)
        combined = torch.empty((output.size(0) * 2, output.size(1), output.size(2), output.size(3)), dtype=image.dtype)
        # left = right = top = bottom = (512 - 128) // 2
        # Pad the image
        # padded_image = TF.pad(image, padding=(left, top, right, bottom), padding_mode='constant', fill=0)
        padded_image = TF.resize(image, size=(64, 64))
        combined[0::2] = padded_image.detach().cpu()
        combined[1::2] = output.detach().cpu()

        grid = make_grid(combined, nrow=2)

        # Save the grid of images
        save_image(grid, f'./saved_images/test_batch_nsr_{batch_idx}_grid.png')        
            # log results and metrics
        results += [output[i].detach().cpu().numpy() for i in range(output.shape[0])]
        targets += [padded_image[i].detach().cpu().numpy() for i in range(output.shape[0])]
        if batch_idx > 4:
            break


    psnr_all = 0
    pcc_all = 0
    ssim_all = 0
    len_data = len(results)
    for result, target in zip(results, targets):
        sample_pcc = pcc(target, result)
        pcc_all += sample_pcc
        sample_psnr = psnr(target, result)
        psnr_all += sample_psnr
        sample_ssim = ssim(target, result) 
        ssim_all += sample_ssim

    print('#########################')    
    print('##Final metric results:##')
    print('#########################')    
    print('PSNR MEAN: {}'.format(psnr_all/len_data))
    print('SSIM MEAN: {}'.format(ssim_all/len_data))
    print('PCC MEAN: {}'.format(pcc_all/len_data))
