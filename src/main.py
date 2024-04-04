import models

def main():
    """Inference the first 10 fMRI with trained networks"""
    # load test fMRI

    # load networks

    # inference
    pass


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


if __name__ == '__main__':
    main()
