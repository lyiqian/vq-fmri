import abc

from data import FMRI, Image, ImageDataset


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
