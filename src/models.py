import abc
import itertools

from data import FMRI, ImNetImage, ImageDataset

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import functional as visionF


## intermediate data classes used by models below
# maybe some of them could just be native python dicts etc.
class SpatialFeats:
    width: int
    height: int
    depth: int


class CodeBook:
    depth: int


class SpatialTokens:
    width: int
    height: int


class NoiseTable:
    pass


class VisualCues:
    tokens: SpatialTokens
    noise_table: NoiseTable


class EncodedVisualCues:
    pass


## model abstract classes
class UNetAbc(abc.ABC):  # Bahman
    @abc.abstractmethod
    def transform(self, *args):
        pass


class FMRIEncoderAbc(abc.ABC):  # TODO TBD
    @abc.abstractmethod
    def encode(self, fmri: FMRI) -> SpatialFeats:
        pass


class ImageEncoderAbc(abc.ABC):  # Eason
    @abc.abstractmethod
    def encode(self, img: ImNetImage) -> SpatialFeats:
        pass


class ImageDecoderAbc(abc.ABC):  # Eason
    @abc.abstractmethod
    def decode(self, spatial_tokens: SpatialTokens) -> ImNetImage:
        pass


class VectorQuantizerAbc(abc.ABC):  # Bahman
    codebook_: CodeBook = None

    @abc.abstractmethod
    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        pass


class VqVaeAbc(abc.ABC):  # Eason
    encoder_: ImageEncoderAbc = None
    decoder_: ImageDecoderAbc = None
    quantizer_: VectorQuantizerAbc = None
    # trailing underscore indicates a variable is assigned after fitting

    @abc.abstractmethod
    def fit(self, img_dataset: ImageDataset):
        pass

    @abc.abstractmethod
    def encode(self, img: ImNetImage) -> SpatialTokens:
        pass

    @abc.abstractmethod
    def decode(self, spatial_tokens: SpatialTokens) -> ImNetImage:
        pass

    @abc.abstractmethod
    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        pass


class TokenClassifierAbc(abc.ABC):  # Eason

    @abc.abstractmethod
    def fit(self, spatial_tokens: SpatialTokens, noise_table: NoiseTable):
        pass

    @abc.abstractmethod
    def predict(self, spatial_tokens: SpatialTokens) -> NoiseTable:
        pass


class DenoiserAbc(abc.ABC):  # Eason
    token_clf: TokenClassifierAbc

    @abc.abstractmethod
    def denoise(self, spatial_tokens: SpatialTokens) -> VisualCues:
        pass


class InpaintingNetworkAbc(abc.ABC):  # Bahman
    @abc.abstractmethod
    def encode(self, vis_tokens: VisualCues) -> EncodedVisualCues:
        pass

    @abc.abstractmethod
    def decode(self, encoded_vis_cues: EncodedVisualCues) -> SpatialTokens:
        pass


class SuperResolutionAbc(abc.ABC):  # Bahman
    @abc.abstractmethod
    def transform(self, spatial_tokens: SpatialTokens) -> SpatialTokens:
        pass


## model concrete classes

# TODO either here or separate phases/pipeline module
# - phase 1: VqVaeAbc
# - phase 2: FMRIEncoderAbc
# - phase 3: DenoiserAbc, InpaintingNetworkAbc


class UNetConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=4, stride=1, padding=1
    ) -> None:
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_layer_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding
        )
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = nn.functional.relu(x)
        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x)
        x = nn.functional.relu(x)
        return x


class UNetEnc(nn.Module):

    def __init__(self, in_channels=8, out_channels=8) -> None:
        super().__init__()
        # TODO: I'm not sure about the channel sizes of middle layers, might need to be calculated by
        # a different formula. Change this if the results were not good!
        # ENCODER PART:
        # uncb_d: UNet Conv Block Down sampling

        # TODO: I set the kernel size of bottleneck to 3, because kernelsize 4 just messes up 
        # layer dimensionalities!
        self.uncb_d_1 = UNetConvBlock(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_sampling_1 = nn.MaxPool2d(2, 2)
        self.uncb_d_2 = UNetConvBlock(
            in_channels=in_channels * 2,
            out_channels=in_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_sampling_2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        uncb_d_1 = self.uncb_d_1(x)
        down_sampling_1 = self.down_sampling_1(uncb_d_1)
        uncb_d_2 = self.uncb_d_2(down_sampling_1)
        down_sampling_2 = self.down_sampling_2(uncb_d_2)
        resids = [uncb_d_1, uncb_d_2]
        return down_sampling_2, resids


class UNetDec(nn.Module):
    def __init__(self, in_channels=8, out_channels=8) -> None:
        super().__init__()
        # DECODER PART:
        self.up_sampling_1 = nn.ConvTranspose2d(
            in_channels=in_channels * 8,
            out_channels=in_channels * 4,
            kernel_size=2,
            stride=2
        )
        self.uncb_u_1 = UNetConvBlock(
            in_channels=in_channels * 8,
            out_channels=in_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.up_sampling_2 = nn.ConvTranspose2d(
            in_channels=in_channels * 4,
            out_channels=in_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.uncb_u_2 = UNetConvBlock(
            in_channels=in_channels * 4,
            out_channels=in_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, resids):
        uncb_d_1, uncb_d_2 = resids
        up_sampling_1 = self.up_sampling_1(x)
        up_sampling_1 = torch.cat([up_sampling_1, uncb_d_2], dim=1)
        uncb_u_1 = self.uncb_u_1(up_sampling_1)
        up_sampling_2 = self.up_sampling_2(uncb_u_1)
        up_sampling_2 = torch.cat([up_sampling_2, uncb_d_1], dim=1)
        uncb_u_2 = self.uncb_u_2(up_sampling_2)
        return uncb_u_2


class UNetDec2X(nn.Module):
    """
    Unet Decoder Used for super resolution, has one more upsampling layer!
    TODO: might be able to write UNet more flexible so that we don't need a
    separate module for different UNet sizes
    """

    def __init__(self, in_channels=8, out_channels=8) -> None:
        super().__init__()
        self.uncb_u_1 = UNetConvBlock(
            in_channels=in_channels * 8,
            out_channels=in_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_sampling_1 = nn.ConvTranspose2d(
            in_channels=in_channels * 8,
            out_channels=in_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.uncb_u_2 = UNetConvBlock(
            in_channels=in_channels * 4,
            out_channels=in_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_sampling_2 = nn.ConvTranspose2d(
            in_channels=in_channels * 4,
            out_channels=in_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.uncb_u_3 = UNetConvBlock(
            in_channels=in_channels * 2,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_sampling_3 = nn.ConvTranspose2d(
            in_channels=in_channels * 1,
            out_channels=in_channels,
            kernel_size=2,
            stride=2,
        )

        def forward(self, x, resids):
            uncb_d_1, uncb_d_2 = resids
            uncb_u_1 = self.uncb_u_1(x)
            uncb_u_1 = torch.cat(uncb_u_1, uncb_d_2)
            up_sampling_1 = self.up_sampling_1(uncb_u_1)
            uncb_u_2 = self.uncb_u_2(up_sampling_1)
            uncb_u_2 = torch.cat(uncb_u_2, uncb_d_1)
            up_sampling_2 = self.up_sampling_2(uncb_u_2)
            # X2 part:
            uncb_u_3 = self.uncb_u_3(up_sampling_2)
            uncb_u_3 = torch.cat(uncb_u_3, uncb_d_1)
            up_sampling_3 = self.up_sampling_3(uncb_u_3)
            return up_sampling_3


class UNet(nn.Module, UNetAbc):  # Bahman

    def __init__(self, in_channels=8, out_channels=8) -> None:
        super().__init__()
        self.enc = UNetEnc(in_channels=in_channels, out_channels=out_channels)
        # BOTTLENECK
        # TODO: I set the kernel size of bottleneck to 3, because kernelsize 4 just messes up 
        # layer dimensionalities!
        self.bottleneck = UNetConvBlock(
            in_channels=in_channels * 4,
            out_channels=in_channels * 8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.dec = UNetDec(in_channels=in_channels, out_channels=out_channels)
        self.unary_conv = nn.Conv2d(
            in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # TODO: I'm also not sure if people still use sigmoid as the last activation function for
        # UNets.
        down_samped, [uncb_d_1, uncb_d_2] = self.enc(x)
        bottleneck = self.bottleneck(down_samped)
        up_samped = self.dec(bottleneck, [uncb_d_1, uncb_d_2])
        unary_conv = torch.sigmoid(self.unary_conv(up_samped))
        return unary_conv

    def transform(self, *args):
        pass


## model abstract classes
class SRNetwork(nn.Module, UNetAbc):  # Bahman

    def __init__(self, in_channels=8, out_channels=8) -> None:
        super().__init__()
        self.enc = UNetEnc(in_channels=in_channels, out_channels=out_channels)
        # BOTTLENECK
        self.bottleneck = UNetConvBlock(
            in_channels=in_channels * 4,
            out_channels=in_channels * 8,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec = UNetDec2X(in_channels=in_channels, out_channels=out_channels)
        self.unary_conv = nn.Conv2d(
            in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # TODO: I'm also not sure if people still use sigmoid as the last activation function for
        # UNets.
        down_samped, uncb_d_1, uncb_d_2 = self.enc(x)
        bottleneck = self.bottleneck(down_samped)
        up_samped = self.dec(bottleneck, uncb_d_1, uncb_d_2)
        unary_conv = torch.sigmoid(self.unary_conv(up_samped))
        return unary_conv


class VectorQuantizer(nn.Module, VectorQuantizerAbc):
    """
    Implemented based on this paper:
        https://arxiv.org/abs/1711.00937.pdf
    The graident part is supposed to be based on this paper,
    tho I implemented it based on what I gathered from the
    original paper.
        https://arxiv.org/pdf/1308.3432.pdf
    """

    def __init__(self, dim_encodings, num_encodings) -> None:
        super().__init__()
        self.dim_encodings = dim_encodings
        self.num_encodings = num_encodings
        self.shape_encodings = [num_encodings, dim_encodings]
        # In some implementations of VQ-VAE, there is a limit on the size of the initial value of the
        # codebook elements. Couldn't find a mention of this in the original paper, but it makes sense
        # since the 'volume' of codebook tensors are supposed to be limited.
        # https://github.com/airalcorn2/vqvae-pytorch/blob/021a7d79cbde845dd322bc0b97f37b08230d3cdc/vqvae.py#L173
        self.lim_encodings = 3**0.5
        self.codebook = torch.tensor(
            torch.rand(self.shape_encodings) * self.lim_encodings, requires_grad=True
        )

    def forward(self, x):
        # x will have the shape B, D, W, H
        # We will want to find the closest neighbours while keeping the dimension D:
        flat_encodings = x.permute(0, 2, 3, 1).reshape(-1, self.dim_encodings)
        # flat_Encodigns B * H * W, D
        # clustering part:
        # get pairwise distances:
        encoding_distances = torch.cdist(flat_encodings, self.codebook)
        # distances.shape = B * H * W, self.num_encodings
        # find closest codebook vector for each encoding output
        encoding_indices = encoding_distances.argmin(dim=1)
        # encoding_indices.shape = B * H * W
        # replace every vector with its closest neighbour in the codebook:
        encoding_quantized = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.codebook
        )
        # Bring the D dimension back to idx 1:
        encoding_quantized = encoding_quantized.permute(0, 3, 1, 2)
        return encoding_quantized, encoding_indices.view(x.shape[0], -1)

    def backward(self, grad_output):
        """
        The https://arxiv.org/abs/1711.00937.pdf paper does straight through gradient estimator,
        for part 1 of the equation 3, the loss for the second part and the 3rd part will be calculated
        afterwards.
        """
        gradinput = F.hardtanh(grad_output)
        return gradinput
    

    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        return self.forward(spatial_feats)


class InpaintingNetwork(nn.Module, InpaintingNetworkAbc):  # Bahman

    def __init__(self, dim_encodings) -> None:
        super().__init__()
        self.network = UNet(in_channels=dim_encodings, out_channels=dim_encodings)

    def forward(self, x, noise_mask):
        # corrupted_token_indexes.shape = B, D, W, H (binary mask)
        # mask = torch.ones_like(x)
        # print(corrupted_token_indexes)
        # mask[corrupted_token_indexes] = 0
        masked_tokens = torch.einsum('bchw,bhw->bchw',x, noise_mask)
        output = self.network(masked_tokens)
        return output

    def encode(self, vis_tokens: VisualCues) -> EncodedVisualCues:
        pass

    def decode(self, encoded_vis_cues: EncodedVisualCues) -> SpatialTokens:
        pass


class SuperResolutionModule(nn.Module, SuperResolutionAbc):  # Bahman

    def __init__(
        self,
        fmri_encoder,
        image_encoder_small,
        image_encoder_large,
        vector_quantizer_fmri,
        vector_quantizer_img_small,
        vector_quantizer_img_large,
        token_classifier,
        inpainting_network,
        image_decoder_large,
        resize_factor=2,
        img_size=[224, 224],
        dim_encodings = 8
    ) -> None:
        super().__init__()
        self.resize_factor = resize_factor
        self.img_size = img_size
        self.down_scaled_size = [int(d) for d in img_size / resize_factor]
        self.fmri_enc = fmri_encoder
        self.img_enc_s = image_encoder_small
        self.img_enc_l = image_encoder_large
        self.img_dec_l = image_decoder_large
        self.fmri_vq = vector_quantizer_fmri
        self.img_vq_s = vector_quantizer_img_small
        self.img_vq_l = vector_quantizer_img_large
        self.token_classifier = token_classifier
        self.inpainting_network = inpainting_network
        self.sr = SRNetwork(in_channels=dim_encodings, out_channels=8)

    def downsize_image(self, image):
        visionF.resize(image, self.down_scaled_size)

    def forward_train(self, y):
        y_ds = self.downsize_image(y)
        z_ds = self.img_vq_s(self.img_enc_s(y_ds))
        z_sr = self.sr(z_ds)
        z = self.img_vq_l(self.img_enc_l(y))
        y_sr = self.img_dec_l(z_sr)
        y = self.img_dec_l(z)
        return y, z , y_sr, z_sr


    def forward(self, x):
        x_mismatches = self.token_classifier(x)
        x_corrected = self.inpainting_network(x, x_mismatches)
        x_sr = self.sr(x_corrected)
        y_sr = self.img_dec_l(x_sr)
        return y_sr

    def transform(self, spatial_tokens: SpatialTokens) -> SpatialTokens:
        pass


class TokenNoising(nn.Module):
    
    def __init__(self, noise_rate) -> None:
        super().__init__()
        self.noise_rate = noise_rate

    def forward(self, x):
        x_clone = x.clone().detach()
        noise_mask = torch.bernoulli(torch.ones([x_clone.shape[0], *x_clone.shape[2:]]) * (1 - self.noise_rate))
        # print(x_clone.shape, noise_mask.shape)
        x_clone = torch.einsum('bchw,bhw->bchw', x_clone, noise_mask)
        return x_clone, noise_mask

    def set_noise_rate(self, rate):
        self.noise_rate = rate

class ResBlock(nn.Module):
    def __init__(self, in_channels, res_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, res_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(res_channels, in_channels, kernel_size=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = out + residual
        out = F.relu(out)
        return out


EncoderConvBlock = UNetConvBlock

class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_t_1 = nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(3)
        self.conv_t_2 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv_t_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.conv_t_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        return x


class ImageEncoder(ImageEncoderAbc, nn.Module):
    def __init__(self, out_channels) -> None:
        super().__init__()
        self.conv_block = EncoderConvBlock(3, out_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_block(x)
        return x

    def encode(self, x):
        return self.forward(x)


class ImageDecoder(ImageDecoderAbc, nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res_block = ResBlock(in_channels, in_channels)
        self.conv_block = DecoderConvBlock(in_channels)

    def forward(self, x):
        x = self.res_block(x)
        x = self.conv_block(x)
        return x

    def decode(self, x):
        return self.forward(x)


class VqVae(VqVaeAbc):
    CODEBOOK_DIM = 8
    CODEBOOK_SIZE = 128

    LR = 2e-4
    ENCODER_ALPHA = 0.25

    def __init__(self) -> None:
        super().__init__()
        self.encoder_ = ImageEncoder(self.CODEBOOK_DIM)
        self.decoder_ = ImageDecoder(self.CODEBOOK_DIM)
        self.quantizer_ = VectorQuantizer(dim_encodings=self.CODEBOOK_DIM, num_encodings=self.CODEBOOK_SIZE)

    def fit(self, img_dataset: ImageDataset):
        loader = torch.utils.data.DataLoader(img_dataset, batch_size=32)

        params = itertools.chain(
            self.encoder_.parameters(),
            self.quantizer_.parameters(),
            self.decoder_.parameters())
        optimizer = torch.optim.Adam(params, lr=self.LR)

        for epoch in range(100):
            print("Training Epoch", epoch)
            for orig in loader:
                spatial_feats = self.encoder_.encode(orig)
                spatial_tokens, __ = self.quantizer_.quantize(spatial_feats)
                reconstructed = self.decoder_.decode(spatial_tokens)

                loss = (
                    (reconstructed - orig).norm(2)
                    + (spatial_feats.detach() - spatial_tokens).norm(2)
                    + (spatial_tokens.detach() - spatial_feats).norm(2) * self.ENCODER_ALPHA
                )
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

    def encode(self, img) -> SpatialTokens:
        spatial_feats = self.encoder_.encode(img)
        spatial_tokens, token_indices = self.quantize(spatial_feats)
        return spatial_tokens, token_indices

    def decode(self, spatial_tokens: SpatialTokens) -> ImNetImage:
        return self.decoder_.decode(spatial_tokens)

    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        return self.quantizer_.quantize(spatial_feats)


class Denoiser(DenoiserAbc):
    def denoise(self, spatial_tokens: SpatialTokens) -> VisualCues:
        noise_table = self.token_clf.predict(spatial_tokens)
        visual_cues = VisualCues(tokens=spatial_tokens, noise_table=noise_table)
        return visual_cues


# class TokenClassifier(TokenClassifierAbc, nn.Module):
#     """We implemented the image classifier ... using the UNet

#     with 2 downsampling and 2 upsampling layers ..."""

#     LR = 1e-4  # TODO TBD

#     def __init__(self):
#         super().__init__()
#         self.encoder = UNetEnc(in_channels=VqVae.CODEBOOK_DIM, out_channels=VqVae.CODEBOOK_DIM)
#         # TODO do we need a bottleneck here?
#         # Note by Bahman: We do, because decoder expects in_channels * 8 input channels but encoder only provides in_channels * 4
#         # self.bottleneck = UNetConvBlock(
#         #     in_channels=VqVae.CODEBOOK_DIM * 4,
#         #     out_channels=VqVae.CODEBOOK_DIM * 8,
#         #     kernel_size=4,
#         #     stride=2,
#         #     padding=1,
#         # )
#         self.decoder = UNetDec(in_channels=int(VqVae.CODEBOOK_DIM /2), out_channels=1)  # True of False

#     def forward(self, x):
#         down_samped, [uncb_d_1, uncb_d_2] = self.encoder(x)
#         # bn_out = self.bottleneck(down_samped)
#         x = self.decoder(down_samped, [uncb_d_1, uncb_d_2])
#         return x

#     def fit(self, spatial_tokens: SpatialTokens, noise_table: NoiseTable):
#         loader = torch.utils.data.DataLoader(
#             torch.utils.data.TensorDataset(spatial_tokens, noise_table),
#             batch_size=32)

#         loss_fn = torch.nn.BCELoss()
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)

#         for epoch in range(100):
#             print("Training epoch", epoch)
#             for tokens, noise in loader:
#                 pred = self.forward(tokens)
#                 loss = loss_fn(pred, noise)

#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#     def predict(self, spatial_tokens: SpatialTokens) -> NoiseTable:
#         return self.forward(spatial_tokens)

class TokenClassifier(TokenClassifierAbc, nn.Module):
    """We implemented the image classifier ... using the UNet

    with 2 downsampling and 2 upsampling layers ..."""

    LR = 2e-4

    def __init__(self):
        super().__init__()
        self.unet= UNet(in_channels=VqVae.CODEBOOK_DIM, out_channels=1)  # True/False in NoiseTable

    def forward(self, x):
        x = self.unet(x)
        return x

    def fit(self, spatial_tokens: SpatialTokens, noise_table: NoiseTable):
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(spatial_tokens, noise_table),
            batch_size=32)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)

        for epoch in range(100):
            print("Training epoch", epoch)
            for tokens, noise in loader:
                pred = self.forward(tokens)
                loss = loss_fn(pred, noise)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, spatial_tokens: SpatialTokens) -> NoiseTable:
        return self.forward(spatial_tokens)


class MLP(nn.Module):
    """.. through 2 hidden layers outputs a feature map z_*^x
    (constrained to be the same size as the z^y)."""

    def __init__(self, in_dims, out_width):
        super().__init__()

        self.in_dims = in_dims
        self.out_width = out_width  # assuming squares
        self.out_dims = VqVae.CODEBOOK_DIM * self.out_width**2

        self.fc1 = nn.Linear(self.in_dims, self.out_dims)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.out_dims, self.out_dims)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.in_dims)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x.view(-1, VqVae.CODEBOOK_DIM, self.out_width, self.out_width)


class FMRIEncoder(FMRIEncoderAbc, nn.Module):
    def __init__(self, in_dims, out_width):
        super().__init__()
        self.mlp = MLP(in_dims, out_width)
        self.unet = UNet(in_channels=VqVae.CODEBOOK_DIM, out_channels=VqVae.CODEBOOK_DIM)

    def forward(self, x):
        x = self.mlp(x)
        x = self.unet(x)
        return x

    def encode(self, fmri: FMRI) -> SpatialFeats:
        return self.forward(fmri)
