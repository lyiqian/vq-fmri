import abc
from data import FMRI, Image, ImageDataset

# torch imports
import torch.nn as nn 
import torch 


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
class UNetAbc(abc.ABC):  # TODO Bahman
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


class VectorQuantizerAbc(abc.ABC):  # TODO Bahman
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
    

