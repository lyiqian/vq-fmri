import torch

from models import TokenNoising
from losses import lossVQ, lossVQ_MSE
from torch.nn import CrossEntropyLoss


def train_phase1(
    train_loader,
    validation_loader,
    test_loader,
    epochs,
    image_encoder,
    image_decoder,
    quantizer,
    token_classifier,
    token_inpainting,
    optimizer,
    beta,
):
    # TODO: noise rate should be set same as error rate in fMRI, we should look up what that value is!
    # TODO: Maybe increasing the noise rate gradually would be beneficial?
    tk = TokenNoising(noise_rate=0.3)
    ce_loss = CrossEntropyLoss()
    for epoch in range(epochs):
        for i, (fmris, images) in enumerate(train_loader):
            # Encode image
            # Learn Codebook
            # Quantize encodigns
            # Train Token Classifier
            # Train Token Inpaingting
            # Decode image
            # Calculate the losses
            optimizer.zero_grad()
            img_encs = image_encoder(images)
            img_encs_q, img_encs_idxs = quantizer(img_encs)
            noisy_encs, noise_mask = tk(img_encs_q)
            enc_idx_preds = token_classifier(noisy_encs)
            enc_val_preds = token_inpainting(noisy_encs, noise_mask)
            with torch.no_grad():
                _, env_val_preds_q_idxs = quantizer(enc_val_preds)
            img_rec = image_decoder(img_encs_q)
            code_book_loss = lossVQ(images, img_rec, img_encs, img_encs_q, beta)
            idx_pred_loss = ce_loss(enc_idx_preds, noise_mask)
            val_pred_loss = lossVQ_MSE(
                z_x=enc_val_preds,
                z_x_q_idxs=env_val_preds_q_idxs,
                z_y_q=img_encs_q,
                z_y_q_idxs=img_encs_idxs,
            )
            code_book_loss.backward()
            idx_pred_loss.backward()
            val_pred_loss.backward()
            optimizer.step()


def train_phase2():
    pass


def train_phase3():
    pass


def train_general():
    pass
