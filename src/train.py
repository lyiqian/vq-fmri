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
    FMRIEncoderAbc,
    UNet,
    SuperResolutionModule,
    MLP
)
from losses import lossVQ, lossVQ_MSE, lossSR
from models import TokenNoising, TokenClassifier, InpaintingNetwork, VqVae, VqVaeAbc, FMRIEncoderAbc, UNet
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.tensorboard import SummaryWriter

from data import GODLoader, ImageLoader

from typing import Type
import itertools

# # TODO: change this with tqdm once it was included in the requirement.txt
# try:
#     from tqdm import tqdm as tqdm_if_available
# except:
#     def tqdm_if_available(*args):
#         return args



# Pipeline of training:
# Phase 1: Train the image vq-vae
# Phase 2: Inpainting and token classifier 
# Phase 3: Train fmri vq-vqe
# Phase 4: Train SR module


def train_phase1(
    train_loader,
    validation_loader,
    test_loader,
    vq_vae: VqVae,
    epochs: int,
    beta: float,
    optimizer,
):
    # torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter()
    glb_iter = 0

    for epoch in range(epochs):
        for i, images in enumerate(train_loader):
            optimizer.zero_grad()
            # VQVQE part:
            # Encode image
            # Learn Codebook
            # Quantize encodigns
            img_encs = vq_vae.encoder_.encode(images)
            img_encs_q, img_encs_idxs = vq_vae.quantizer_.quantize(img_encs)
            img_rec = vq_vae.decoder_.decode(img_encs_q)
            loss = lossVQ(images, img_rec, img_encs, img_encs_q, beta)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Loss @ Ep{epoch} Batch{i}: {loss.item()}")
                encoded, __ = vq_vae.encode(train_loader.dataset[1].unsqueeze(0))
                decoded = vq_vae.decode(encoded).squeeze(0)
                writer.add_image('phase1/decoded_img', decoded, glb_iter)
            writer.add_scalar('phase1/loss', loss.item(), glb_iter)
            glb_iter += 1

    writer.close()

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
            loss = lossVQ_MSE(fmri_feats, fmri_codebook_idxs, img_tokens, img_codebook_idxs)

            loss.backward()
            optimizer.step()

        print(f"Loss of last batch @ Epoch {ep}: {loss.item()}")


def train_phase3(
    train_loader: GODLoader,
    validation_loader: GODLoader,
    test_loader: GODLoader,
    epochs: int,
    trained_vq_vae: VqVaeAbc,
    token_classifier: TokenClassifier,
    token_inpainting: InpaintingNetwork,
    beta: float,
    lr: float = 0.01,
):
    # TODO: noise rate should be set same as error rate in fMRI, we should look up what that value is!
    # TODO: Maybe increasing the noise rate gradually would be beneficial?

    # Train Token Classifier
    # Train Token Inpaingting
    # Decode image
    # Calculate the losses
    tk = TokenNoising(noise_rate=0.3)
    ce_loss = BCELoss()
    token_classifier_optimizer = torch.optim.Adam(token_classifier.parameters(), lr=lr)
    token_inpainting_optimizer = torch.optim.Adam(token_inpainting.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            token_classifier_optimizer.zero_grad()
            token_inpainting_optimizer.zero_grad()
            img_encs_q, img_encs_idxs = trained_vq_vae.encode(images)
            with torch.no_grad():
                noisy_encs, noise_mask = tk(img_encs_q)
            enc_idx_preds = token_classifier(noisy_encs)
            enc_val_preds = token_inpainting(noisy_encs, noise_mask)
            with torch.no_grad():
                _, env_val_preds_q_idxs = trained_vq_vae.quantizer_.quantize(enc_val_preds)
            # print(enc_idx_preds.shape)
            # print(noise_mask.shape)
            idx_pred_loss = ce_loss(
                enc_idx_preds,
                noise_mask.view(noise_mask.shape[0], 1, *noise_mask.shape[1:]),
            )
            # TODO: double check inputs of this loss:
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




def train_sr(
    image_loader: ImageLoader,
    sr_module: SuperResolutionModule,
    vq_vae_large: VqVae,
    epochs: int,
    beta:float
):
    """Eventhough the main paper divides the training into 3 main phases,
    there is also a 4th phase for training the super resolution modules
    HACK/TODO: the original paper does not mention anything about how the vector quantization of larger VQVQE and down sampled VQVAE are related.
    Here I'm using a completly different VQVAE for larger samples, but we can explore adding the codes from smaller codebook to the larger
    VQVAE codebook set.
    """
    # TODO
    #  - train a larger image encoder decoder
    params = itertools.chain(
        vq_vae_large.encoder_.parameters(),
        vq_vae_large.quantizer_.parameters(),
        vq_vae_large.decoder_.parameters(),
    )
    optimizer = torch.optim.Adam(params, lr=vq_vae_large.LR)
    for _ in range(epochs):
        for i, images in enumerate(image_loader):
            optimizer.zero_grad()
            img_encs = vq_vae_large.encoder_.encode(images)
            img_encs_q, _ = vq_vae_large.quantizer_.quantize(img_encs)
            img_rec = vq_vae_large.decoder_.decode(img_encs_q)
            code_book_loss = lossVQ(images, img_rec, img_encs, img_encs_q, beta)
            code_book_loss.backward()
            optimizer.step()

    # TODO tune lr value
    sr_module.vq_vae_l = vq_vae_large
    optimizer_sr = torch.optim.Adam(sr_module.sr.parameters(), lr=1e-4)
    for _ in range(epochs):
        for i, images in enumerate(image_loader):
            optimizer_sr.zero_grad()
            # y: image
            # z: encoding
            # _q: quantized encoding
            # sr: super resolution network (not to be confused with super resolution module) output
            # l: large (original size)
            # _idx: index of chosen vectors in the codebook, they are needed for finding mismatches in the VQ loss
            y_l, z_l, z_l_q, z_l_q_idx, y_sr, z_sr, z_sr_q, z_sr_q_idx = (
                sr_module.forward_train(images)
            )
            code_book_loss = lossSR(y_l, y_sr, z_l, z_l_q_idx, z_sr, z_sr_q_idx)
            code_book_loss.backward()
            optimizer_sr.step()


def inference(godloader:GODLoader, fmri_mlp: MLP, fmri_vqvae:VqVae, sr_module: SuperResolutionModule):
    for idx_batch, (image, fmri) in enumerate(godloader):
        transfmored_fmri = fmri_mlp(fmri.to(torch.float32)[:,:10])
        encoded_fmri, _ = fmri_vqvae.encode(transfmored_fmri)
        output = sr_module(encoded_fmri)
        print(output.shape)


def train_general():
    data_loader = GODLoader(
        data_dir="/Users/bahman/Documents/courses/Deep Learning/Project/code/data",
        batch_size=2,
    )
    # data_loader = ImageLoader(
    #     data_dir="/Users/bahman/Documents/courses/Deep Learning/Project/code/data",
    #     batch_size=16,
    # )
    train_loader = data_loader.get_train_loader()
    validation_loader = None
    test_loader = None
    epochs = 2
    vq_vqe = VqVae()
    token_classifier = TokenClassifier()
    token_inpainting = InpaintingNetwork(8)
    beta = 2
    # train_phase1(
    #     train_loader=train_loader,
    #     validation_loader=validation_loader,
    #     test_loader=test_loader,
    #     epochs=epochs,
    #     vq_vae=vq_vqe,
    #     token_classifier=token_classifier,
    #     token_inpainting=token_inpainting,
    #     beta=beta,
    # )

    train_loader = data_loader.get_train_loader()
    validation_loader = None
    test_loader = None
    epochs = 2
    vq_vae_large = VqVae()
    vq_vae_s = VqVae()
    vq_vae_fmri = VqVae(in_channels=8)
    token_classifier = TokenClassifier()
    token_inpainting = InpaintingNetwork(8)
    sr_module = SuperResolutionModule(
        vq_vae_s,
        vq_vae_large,
        vq_vae_fmri,
        token_classifier,
        token_inpainting,
        resize_factor=2,
        img_size=[128, 128],
    )
    beta = 2
    # train_sr(train_loader, sr_module, vq_vae_large, epochs, beta)
    fmri_mlp = MLP(10, 32)
    inference(train_loader, fmri_mlp, vq_vae_fmri, sr_module)

train_general()
