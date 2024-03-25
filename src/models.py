import abc
from data import FMRI, Image, ImageDataset

# torch imports
import torch.nn as nn 
import torch 

import torch.nn.functional as F

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


class VisualCues(SpatialTokens):
    pass


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


class ImageEncoderAbc(abc.ABC):  # TODO Eason
    @abc.abstractmethod
    def encode(self, img: Image) -> SpatialFeats:
        pass


class ImageDecoderAbc(abc.ABC):  # TODO Eason
    @abc.abstractmethod
    def decode(self, spatial_tokens: SpatialTokens) -> Image:
        pass


class VectorQuantizerAbc(abc.ABC):  # Bahman
    codebook_: CodeBook = None

    @abc.abstractmethod
    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        pass


class VqVaeAbc(abc.ABC):  # TODO Eason
    encoder_: ImageEncoderAbc = None
    decoder_: ImageDecoderAbc = None
    quantizer_: VectorQuantizerAbc = None
        # trailing underscore indicates a variable is assigned after fitting

    @abc.abstractmethod
    def fit(self, img_dataset: ImageDataset):
        pass

    @abc.abstractmethod
    def encode(self, img: Image) -> SpatialTokens:
        pass

    @abc.abstractmethod
    def decode(self, spatial_tokens: SpatialTokens) -> Image:
        pass

    @abc.abstractmethod
    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        pass


class TokenClassifierAbc(abc.ABC):  # TODO Eason

    @abc.abstractmethod
    def fit(self, spatial_tokens: SpatialTokens, noise_table: NoiseTable):
        pass

    @abc.abstractmethod
    def predict(self, spatial_tokens: SpatialTokens) -> NoiseTable:
        pass


class DenoiserAbc(abc.ABC):  # TODO Eason
    token_clf: TokenClassifierAbc

    @abc.abstractmethod
    def denoise(self, spatial_tokens: SpatialTokens) -> VisualCues:
        pass


class InpaintingNetworkAbc(abc.ABC):  # TODO Bahman
    @abc.abstractmethod
    def encode(self, vis_tokens: VisualCues) -> EncodedVisualCues:
        pass

    @abc.abstractmethod
    def decode(self, encoded_vis_cues: EncodedVisualCues) -> SpatialTokens:
        pass


class SuperResolutionAbc(abc.ABC):  # TODO Bahman
    @abc.abstractmethod
    def transform(self, spatial_tokens: SpatialTokens) -> SpatialTokens:
        pass


## model concrete classes

# TODO either here or separate phases/pipeline module
# - phase 1: VqVaeAbc
# - phase 2: FMRIEncoderAbc
# - phase 3: DenoiserAbc, InpaintingNetworkAbc


class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1) -> None:
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv_layer_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = nn.functional.relu(x)
        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x)
        x = nn.functional.relu(x)
        return x
        

## model abstract classes
class UNet(nn.Module, UNetAbc):  # TODO Bahman

    def __init__(self, in_channels=8, out_channels=8) -> None:
        super().__init__()
        # TODO: I'm not sure about the channel sizes of middle layers, might need to be calculated by
        # a different formula. Change this if the results were not good!
        # ENCODER PART:
        # uncb_d: UNet Conv Block Down sampling
        self.uncb_d_1 = UNetConvBlock(in_channels=in_channels, out_channels=in_channels*2, kernel_size=4, stride=2, padding=1)
        self.down_sampling_1 = nn.MaxPool2d(2, 2)
        self.uncb_d_2 = UNetConvBlock(in_channels=in_channels*2, out_channels=in_channels*4, kernel_size=4, stride=2, padding=1)
        self.down_sampling_2 = nn.MaxPool2d(2, 2)
        # BOTTLENECK
        self.bottleneck = UNetConvBlock(in_channels=in_channels*4, out_channels=in_channels*8, kernel_size=4, stride=2, padding=1)
        # DECODER PART:
        self.uncb_u_1 = UNetConvBlock(in_channels=in_channels*8, out_channels=in_channels*4, kernel_size=4, stride=2, padding=1)
        self.up_sampling_1 = nn.ConvTranspose2d(in_channels=in_channels*8, out_channels=in_channels*4, kernel_size=2, stride=2)   
        self.uncb_u_2 = UNetConvBlock(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=4, stride=2, padding=1)
        self.up_sampling_2 = nn.ConvTranspose2d(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=2, stride=2)   
        self.unary_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=8, kernel_size=1)
    

    def forward(self, x):
        # TODO: I'm also not sure if people still use sigmoid as the last activation function for 
        # UNets.
        uncb_d_1 = self.uncb_d_1(x)
        down_sampling_1 = self.down_sampling_1(uncb_d_1)
        uncb_d_2 = self.uncb_d_2(down_sampling_1)
        down_sampling_2 = self.down_sampling_2(uncb_d_2)
        bottleneck = self.bottleneck(down_sampling_2)
        uncb_u_1 = self.uncb_u_1(bottleneck)
        uncb_u_1 = torch.cat(uncb_u_1, uncb_d_2)
        up_sampling_1 = self.up_sampling_1(uncb_u_1)
        uncb_u_2 = self.uncb_u_2(up_sampling_1)
        uncb_u_2 = torch.cat(uncb_u_2, uncb_d_1)
        up_sampling_2 = self.up_sampling_2(uncb_u_2)
        unary_conv = torch.sigmoid(self.unary_conv(up_sampling_2))
        return unary_conv

    def transform(self, *args):
        pass
    



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
        self.lim_encodings = 3 ** 0.5 
        self.codebook = torch.tensor(torch.rand(self.shape_encodings) * self.lim_encodings, requires_grad=True)

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
        encoding_quantized = F.embedding(encoding_indices.view(x.shape[0], x.shape[2:]), self.codebook)
        # Bring the D dimension back to idx 1:
        encoding_quantized = encoding_quantized.permute(0, 3, 1, 2)
        return encoding_quantized
    

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