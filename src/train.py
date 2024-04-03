import torch

from models import TokenNoising, ImageEncoder, ImageDecoder, VectorQuantizer, TokenClassifier, InpaintingNetwork, VqVae, VqVaeAbc, FMRIEncoderAbc
from losses import lossVQ, lossVQ_MSE
from torch.nn import CrossEntropyLoss

from data import GODDataset, GODLoader

from typing import Type
import itertools

def train_phase1(
    train_loader: GODDataset,
    validation_loader: GODDataset,
    test_loader: GODDataset,
    epochs: int,
    vq_vae: VqVae,
    token_classifier: TokenClassifier,
    token_inpainting: InpaintingNetwork,
    beta: float,
    lr:float=0.01
):
    # TODO: noise rate should be set same as error rate in fMRI, we should look up what that value is!
    # TODO: Maybe increasing the noise rate gradually would be beneficial?


    params = itertools.chain(
        vq_vae.encoder_.parameters(),
        vq_vae.quantizer_.parameters(),
        vq_vae.decoder_.parameters())
    optimizer = torch.optim.Adam(params, lr=vq_vae.LR)    

    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()
            # VQVQE part:
            # Encode image
            # Learn Codebook
            # Quantize encodigns
            img_encs = vq_vae.encoder_.encode(images)
            img_encs_q, img_encs_idxs = vq_vae.quantizer_.quantize(img_encs)
            img_rec = vq_vae.decoder_.decode(img_encs_q)
            code_book_loss = lossVQ(images, img_rec, img_encs, img_encs_q, beta)
            code_book_loss.backward()
            optimizer.step()

    # Train Token Classifier
    # Train Token Inpaingting
    # Decode image
    # Calculate the losses
    tk = TokenNoising(noise_rate=0.3)
    ce_loss = CrossEntropyLoss()
    token_classifier_optimizer = torch.optim.Adam(token_classifier.parameters(), lr=lr)    
    token_inpainting_optimizer = torch.optim.Adam(token_inpainting.parameters(), lr=lr)    

    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            token_classifier_optimizer.zero_grad()
            token_inpainting_optimizer.zero_grad()
            img_encs_q = vq_vae.encode(images)
            noisy_encs, noise_mask = tk(img_encs_q)
            enc_idx_preds = token_classifier(noisy_encs)
            enc_val_preds = token_inpainting(noisy_encs, noise_mask)
            with torch.no_grad():
                _, env_val_preds_q_idxs = vq_vae.quantizer_.quantize(enc_val_preds)
            idx_pred_loss = ce_loss(enc_idx_preds, noise_mask)
            val_pred_loss = lossVQ_MSE(
                z_x=enc_val_preds,
                z_x_q_idxs=env_val_preds_q_idxs,
                z_y_q=img_encs_q,
                z_y_q_idxs=img_encs_idxs,
            )
            idx_pred_loss.backward()
            val_pred_loss.backward()
            token_classifier_optimizer.step()
            token_inpainting_optimizer.step()


def train_phase2(
    train_loader,
    epochs,
    fmri_encoder: FMRIEncoderAbc,
    trained_vq_vae: VqVaeAbc,
):
    optimizer = torch.optim.Adam(fmri_encoder.parameters(), lr=2e-4)

    for ep in range(epochs):
        for images, fmris in train_loader:
            optimizer.zero_grad()

            fmri_feats = fmri_encoder(fmris)
            fmri_tokens, fmri_codebook_idxs = trained_vq_vae.quantize(fmri_feats)
            img_tokens, img_codebook_idxs = trained_vq_vae.encode(images)
            loss = lossVQ(fmri_feats, fmri_codebook_idxs, img_tokens, img_codebook_idxs)

            loss.backward()
            optimizer.step()

        print(f"Loss of last batch @ Epoch {ep}: {loss.item()}")


def train_phase3():
    pass


def train_general():
    data_loader = GODLoader(data_dir='/Users/bahman/Documents/courses/Deep Learning/Project/code/data', batch_size=16)
    train_loader = data_loader.get_train_loader()
    validation_loader = None
    test_loader = None
    epochs = 2
    vq_vqe = VqVae()
    token_classifier = TokenClassifier()
    token_inpainting = InpaintingNetwork(8)
    beta = 2
    train_phase1(train_loader=train_loader, validation_loader=validation_loader,test_loader=test_loader, epochs=epochs,
                 vq_vae=vq_vqe, token_classifier=token_classifier, token_inpainting=token_inpainting, beta=beta)



train_general()