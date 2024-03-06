## data classes
class SpatialFeats:
    pass


class CodeBook:
    pass


class SpatialTokens:
    pass


class NoiseTable:
    pass


class VisualTokens(SpatialTokens):
    pass


class EncodedVisualTokens:
    pass


## model classes
class UNet:
    pass  # TODO needed in 3? places


class FMRIEncoder:
    # TODO see p4 Discrete Visual Cues.
    ...

    def encode(fmri: FMRI) -> SpatialFeats:
        pass


class ImageEncoder:
    def encode(img: Image) -> SpatialFeats:
        pass


class ImageDecoder:
    def decode(SpatialTokens) -> Image:
        pass


class VectorQuantizer:
    def __init__(codebook: CodeBook):
        pass

    def quantize(spatial_feats: SpatialFeats) -> SpatialTokens:
        pass


class VqVae:
    ImageEncoder
    ImageDecoder
    VectorQuantizer
    CodeBook

    def fit(ImageDataset):
        pass

    def quantize(spatial_feats: SpatialFeats) -> SpatialTokens:
        pass

    def encode(img: Image) -> SpatialTokens:
        pass

    def decode(spatial_tokens: SpatialTokens) -> Image:
        pass


class TokenClassifier:
    def fit(SpatialTokens, NoiseTable):
        pass

    def predict(SpatialTokens) -> NoiseTable:
        pass


class Denoiser:
    TokenClassifier

    def denoise(SpatialTokens) -> VisualTokens:
        pass


class InpaintingEncoder:
    def encode(vis_tokens: VisualTokens) -> EncodedVisualTokens:
        pass


class InpaintingDecoder:
    def decode(EncodedVisualTokens) -> SpatialTokens:
        pass


class SuperResolution:
    pass  # TODO figure 4
