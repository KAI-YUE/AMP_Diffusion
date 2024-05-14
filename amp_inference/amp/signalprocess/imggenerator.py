import numpy as np
import torch

class ImageGenerator(object):
    def __init__(self, dataloader, **kwargs):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

        self.config = kwargs.get("config")

    def generate(self):
        # extract batch
        data_batch = next(self.iterator)

        (x, label) = data_batch

        # flatten the signal to a vector
        if self.config.signal_type == "grey":
            x_numpy = greytensor_2_npvec(x, self.config.source_bound)
        else:
            x_numpy = rgbtensor_2_npvec(x, self.config.source_bound, to_grey=False)

        return x_numpy
    

def greytensor_2_npvec(x, source_bound=255):
    x_numpy = x.flatten()
    x_numpy = x_numpy.numpy().reshape(-1,1)
    return source_bound*x_numpy


def rgbtensor_2_npvec(x, source_bound=255, to_grey=True):
    # Convert tensor back to numpy array
    # if grey, then it is a Nx1 array
    # if RGB, then it is a Nx3 array
    x = x.cpu().detach().numpy()

    if to_grey:
        x = x[0, 0, ...]
        x_vec = source_bound*x.reshape(-1, 1)
    else:
        x_r, x_g, x_b = x[0, 0, ...], x[0, 1, ...], x[0, 2, ...]
        x_r_vec, x_g_vec, x_b_vec = x_r.reshape(-1, 1), x_g.reshape(-1, 1), x_b.reshape(-1, 1)
        x_r_vec, x_g_vec, x_b_vec = source_bound*x_r_vec, source_bound*x_g_vec, source_bound*x_b_vec
        # concatenate the three channels into a matrix
        x_vec = np.hstack((x_r_vec, x_g_vec, x_b_vec))

    return x_vec


def npvec2tensor(y, im_size, source_ch=1, batch_size=1):
    
    if source_ch == 1:
        # Reshape to greyscale image of shape (1, ch, im_size, im_size)
        # Replicate the greyscale channel to mimic RGB format
        array_reshaped = y.reshape(batch_size, -1, im_size[0], im_size[1])
        array_rgb = np.repeat(array_reshaped, 3, axis=1)
    else:
        array_rgb = np.zeros((batch_size, 3, im_size[0], im_size[1]))
        for i in range(y.shape[1]):
            array_rgb[:, i, :, :] = y[:, i].reshape(im_size[0], im_size[1])

    # Convert to torch tensor
    array_rgb_tensor = torch.from_numpy(array_rgb/255).float()
    return array_rgb_tensor
