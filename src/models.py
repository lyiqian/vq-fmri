import abc

from data import FMRI, Image, ImageDataset


## intermediate data classes used by models below
# maybe some of them could be native python dicts etc.
class SpatialFeats:
    pass


class CodeBook:
    pass


class SpatialTokens:
    pass


class NoiseTable:
    pass


class VisualCues(SpatialTokens):
    pass


class EncodedVisualCues:
    pass


## model abstract classes
class UNetAbc(abc.ABC):
    @abc.abstractmethod
    def transform(self, *args):
        pass


class FMRIEncoderAbc(abc.ABC):
    @abc.abstractmethod
    def encode(self, fmri: FMRI) -> SpatialFeats:
        pass


class ImageEncoderAbc(abc.ABC):
    @abc.abstractmethod
    def encode(self, img: Image) -> SpatialFeats:
        pass


class ImageDecoderAbc(abc.ABC):
    @abc.abstractmethod
    def decode(self, spatial_tokens: SpatialTokens) -> Image:
        pass


class VectorQuantizerAbc(abc.ABC):
    def __init__(self, codebook: CodeBook):
        self.codebook = codebook

    @abc.abstractmethod
    def quantize(self, spatial_feats: SpatialFeats) -> SpatialTokens:
        pass


class VqVaeAbc(abc.ABC):
    codebook: CodeBook
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


class TokenClassifierAbc(abc.ABC):

    @abc.abstractmethod
    def fit(self, spatial_tokens: SpatialTokens, noise_table: NoiseTable):
        pass

    @abc.abstractmethod
    def predict(self, spatial_tokens: SpatialTokens) -> NoiseTable:
        pass


class DenoiserAbc(abc.ABC):
    token_clf: TokenClassifierAbc

    @abc.abstractmethod
    def denoise(self, spatial_tokens: SpatialTokens) -> VisualCues:
        pass


class InpaintingNetworkAbc(abc.ABC):
    @abc.abstractmethod
    def encode(self, vis_tokens: VisualCues) -> EncodedVisualCues:
        pass

    @abc.abstractmethod
    def decode(self, encoded_vis_cues: EncodedVisualCues) -> SpatialTokens:
        pass


class SuperResolutionAbc(abc.ABC):
    @abc.abstractmethod
    def transform(self, spatial_tokens: SpatialTokens) -> SpatialTokens:
        pass


## model concrete classes
# TODO either here or separate phases/pipeline module
