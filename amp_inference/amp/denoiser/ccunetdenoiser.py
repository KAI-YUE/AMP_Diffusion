import torch
import torch.jit
import numpy as np

from amp.signalprocess import greytensor_2_npvec, rgbtensor_2_npvec, npvec2tensor

class CCUnetDenoiser(object):
    def __init__(self, config, denoiser_type='ccunet'):
        self.im_size = config.im_size
        self.device = config.device

        self.denoiser = denoiser_type
        self.signal_source_channel = config.channel # the channel of the signal source
        self.bound = 150  # the sigma bound for [0, 255] images
        self.channel = 3  # RGB as the input to the network

        self.source_bound = config.source_bound
        self.prepare_nn(config)
        self.signal_type = config.signal_type

    def prepare_nn(self, config):
        self.model = torch.jit.load(config.checkpoint)

        self.model = self.model.to(config.device)
        self.model.eval()
        # self.model.requires_grad_(False)

    def __call__(self, y, sigma_hat):        
        rgb_tensor = npvec2tensor(y, self.im_size, self.signal_source_channel)
        rgb_tensor = rgb_tensor.to(self.device)

        # Denoise
        sigma = np.clip(sigma_hat, 0, self.bound-1)
        sigma_tensor = torch.tensor(sigma, dtype=torch.int).view(1, -1)
        sigma_tensor = sigma_tensor.to(self.device)
        output = self.model(rgb_tensor, sigma_tensor)
        
        to_grey = True if self.signal_type == "grey" else False
        output_vec = rgbtensor_2_npvec(output, self.source_bound, to_grey=to_grey)

        return output_vec
    
