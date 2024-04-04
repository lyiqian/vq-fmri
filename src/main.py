import models

def main():
    pass


def inference(
    fmri,
    vq_vae: models.VqVaeAbc,
    fmri_enc: models.FMRIEncoderAbc,
    token_clf: models.TokenClassifierAbc,
    inpainting: models.InpaintingNetworkAbc,
):
    spatial_feats = fmri_enc.encode(fmri)

    spatial_token = vq_vae.quantize(spatial_feats)
    noise_table = token_clf.predict(spatial_token)

    visual_cues = inpainting(spatial_token, noise_table)

    reconstructed_img = vq_vae.decode(visual_cues)
    return reconstructed_img


if __name__ == '__main__':
    main()
