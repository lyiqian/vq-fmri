import argparse

import matplotlib.pyplot as plt
import torch

from data import GODLoader, ImageLoader
from train import train_phase1, train_phase2, train_phase3, train_sr
from inference import inference
import models
from pathlib import Path

# for reproducability:
import torch

torch.manual_seed(0)


def main():
    """Inference the first 10 fMRI with trained networks"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, default="../data/GOD")
    parser.add_argument("-w", "--weight-dir", type=str, default="../weights/")
    parser.add_argument("-v", "--vq", type=str, default="vq")
    parser.add_argument("-f", "--fmri-enc", type=str, default="fmri_enc")
    parser.add_argument("-c", "--token-clf", type=str, default="token_clf")
    parser.add_argument("-i", "--inpainting", type=str, default="inpainting")
    parser.add_argument("-s", "--super-res", type=str, default="super_res")
    parser.add_argument("-r", "--img-dec", type=str, default="img_dec")
    parser.add_argument("-l", "--log_dir", type=str, default="../logs/")
    parser.add_argument("-m", "--model_dir", type=str, default="../models/")
    parser.add_argument("-p", "--phase", type=str, default="phase1")
    parser.add_argument("-e", "--resume", type=int, default=0)
    args = parser.parse_args()

    # load test fMRI
    # fmri_ds = data.GODDataset(args.data_dir, image_transforms=None, split='test')
    # fmri = torch.stack([f for __, f in fmri_ds])
    if args.phase == "phase1":
        image_loader = ImageLoader(data_dir=args.data_dir, batch_size=16, image_size=128)
        train_loader = image_loader.get_train_loader()
        vq_vae = models.VqVae()
        epochs = 50
        beta = 1
        # train_phase2(image_loader, epochs, vq_vae, )
        train_phase1(
            train_loader,
            None,
            None,
            vq_vae,
            epochs,
            beta,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
        )
    if args.phase == "phase3":
        trained_vq_vae = models.VqVae()
        trained_vq_vae.load(args.model_dir, 4)
        epochs = 50
        beta = 1
        image_loader = ImageLoader(data_dir=args.data_dir, batch_size=16, image_size=128)
        train_loader = image_loader.get_train_loader()
        token_classifier = models.TokenClassifier()
        token_classifier.train()
        token_inpainting = models.InpaintingNetwork(trained_vq_vae.CODEBOOK_DIM)
        token_inpainting.train()
        lr = 1e-3
        train_phase3(
            train_loader,
            epochs,
            trained_vq_vae,
            token_classifier,
            token_inpainting,
            beta,
            lr,
            args.log_dir,
            args.model_dir,
            resume_from=args.resume
        )
    if args.phase == "sr":
        image_loader = ImageLoader(
            data_dir=args.data_dir, batch_size=16, image_size=256
        )
        image_loader = image_loader.get_train_loader()
        vq_vae_s = models.VqVae()
        vq_vae_s.load(args.model_dir, 4)
        vq_vae_s.train()
        vq_vae_l = models.VqVae(codebook_size=256)
        # We can get away with not having any token classifier, inpainting or fmri moudles
        # in the training phase since we don't need them in the training phase
        sr_module = models.SuperResolutionModule(
            vq_vae_s=vq_vae_s,
            vq_vae_l=vq_vae_l,
            vq_vae_fmri=None,
            token_classifier=None,
            inpainting_network=None,
            resize_factor=2,
            img_size=[256, 256],
        )
        epochs = 4
        beta = 1
        train_sr(
            image_loader,
            sr_module,
            vq_vae_l,
            epochs,
            beta,
            args.log_dir,
            args.model_dir,
            resume_from_ph1=args.resume
        )
    if args.phase == "inf":
        god_loader = GODLoader(data_dir=args.data_dir, batch_size=32)
        test_loader = god_loader.get_test_loader()
        inference(test_loader)


    # # load networks
    # vq = models.VectorQuantizer()
    # fmri_enc = models.FMRIEncoder()
    # token_clf = models.TokenClassifier()
    # inpainting = models.InpaintingNetwork()
    # sr = models.SuperResolutionModule()
    # img_dec = models.ImageDecoder()
    # vq.load_state_dict(torch.load(f'{args.weight_dir}/{args.vq}.pth'))
    # fmri_enc.load_state_dict(torch.load(f'{args.weight_dir}/{args.fmri_enc}.pth'))
    # token_clf.load_state_dict(torch.load(f'{args.weight_dir}/{args.token_clf}.pth'))
    # inpainting.load_state_dict(torch.load(f'{args.weight_dir}/{args.inpainting}.pth'))
    # sr.load_state_dict(torch.load(f'{args.weight_dir}/{args.super_res}.pth'))
    # img_dec.load_state_dict(torch.load(f'{args.weight_dir}/{args.img_dec}.pth'))

    # # fMRI to image
    # reconstructed_img = decode_fmri(
    #     fmri, vq, fmri_enc, token_clf, inpainting, sr, img_dec,
    # )

    # # display a sample
    # plt.imshow(reconstructed_img[0].permute(1, 2, 0))
    # plt.show()


def decode_fmri(
    fmri,
    vq_l: models.VectorQuantizerAbc,
    fmri_enc: models.FMRIEncoderAbc,
    token_clf: models.TokenClassifierAbc,
    inpainting: models.InpaintingNetworkAbc,
    sr: models.SuperResolutionAbc,
    img_dec: models.ImageDecoderAbc,
):
    spatial_feats = fmri_enc.encode(fmri)

    spatial_token_l = vq_l.quantize(spatial_feats)
    noise_table = token_clf.predict(spatial_token_l)

    visual_cues_l = inpainting(spatial_token_l, noise_table)
    visual_cues = sr.transform(visual_cues_l)

    reconstructed_img = img_dec.decode(visual_cues)
    return reconstructed_img


if __name__ == "__main__":
    main()
