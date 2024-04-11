import torch 
from torch.nn.functional import mse_loss


def lossVQ(img, img_rec, encs, cb_codes, beta):
    """
        formula 2 in the vq-fmri paper or formula 3 in the vq-vae paper
    """
    reconstruction_loss = mse_loss(img, img_rec)
    with torch.no_grad():
        cb_codes_clone = cb_codes.clone().detach()
        encs_clone = encs.clone().detach()
    commitment_loss = mse_loss(encs, cb_codes_clone)
    codebook_loss = mse_loss(encs_clone, cb_codes)
    vq_loss = 20 * reconstruction_loss + codebook_loss + beta * commitment_loss
    return vq_loss

def lossVQ_MSE(z_x, z_x_q_idxs, z_y_q, z_y_q_idxs):
    """ Formula 5 in the paper: the formula basically tries to bring two sets of learned quantized codes
    closer together without regressing to the mean

    Args:
        z_x (_type_): first set of encodings 
        z_x_q_idxs (_type_): indexes of codebook vectors after quantization of 1st set (for easier comparison)
        z_y_q (_type_): the quantization vectors of the second set of encodings
        z_y_q_idxs (_type_): indexes of codebook vectors after quantization of 2nd set (for easier comparison)
    """
    with torch.no_grad():
        mismatches = torch.ne(z_x_q_idxs, z_y_q_idxs).int().view([z_x.shape[0], *z_x.shape[2:]])
    ml = mse_loss(z_x, z_y_q, reduction='none')
    loss = torch.einsum('bchw,bhw->bchw', ml, mismatches).mean()
    return loss 


def lossSR(y, rec_y, z_x, z_x_q_idxs, z_y_q, z_y_q_idxs):
    loss_vq_mse = lossVQ_MSE(z_x, z_x_q_idxs, z_y_q, z_y_q_idxs)
    loss_rec  = (y - rec_y).norm(p=2)
    return loss_vq_mse + loss_rec