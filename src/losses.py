import torch 
from torch.nn.functional import mse_loss


def lossVQ(img, img_rec, encs, cb_codes, beta):
    """
        formula 2 in the vq-fmri paper or formula 3 in the vq-vae paper
    """
    reconstruction_loss = mse_loss(img, img_rec)
    cb_codes_clone = cb_codes.clone().detach()
    encs_clone = encs.clone().detach()
    commitment_loss = mse_loss(encs, cb_codes_clone)
    codebook_loss = mse_loss(encs_clone, cb_codes)
    vq_loss = reconstruction_loss + codebook_loss + beta * commitment_loss
    return vq_loss

def lossVQ_MSE(z_x, z_x_q_idxs, z_y_q, z_y_q_idxs):
    """ Formula 5 in the paper: the formula basically tries to bring to sets of learned quantized codes
    closer together without regressing to the mean

    Args:
        z_x (_type_): first set of encodings 
        z_x_q_idxs (_type_): indexes of codebook vectors after quantization of 1st set (for easier comparison)
        z_y_q (_type_): the quantization vectors of the second set of encodings
        z_y_q_idxs (_type_): indexes of codebook vectors after quantization of 2nd set (for easier comparison)
    """
    with torch.no_grad():
        mismatches = 1 - (z_x_q_idxs == z_y_q_idxs).int()
    loss = (mse_loss((z_x, z_y_q), reduction='none') * mismatches).mean()
    return loss 
