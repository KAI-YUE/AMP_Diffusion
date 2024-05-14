import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve

class GaussianDenoiser(object):
    def __init__(self, config):
        self.im_size = config.im_size
        self.signal_type = config.signal_type

    # def __call__(self, y, sigma_hat):
    #     y_img = y.reshape(self.im_size)

    #     output = gaussian_filter(np.uint8(255*y_img), sigma=sigma_hat)
    #     output_vec = output.reshape(-1, 1)

    #     return output_vec
        
    def __call__(self, y, sigma_hat):
        
        h = matlab_style_gauss2D(shape=(5,5), sigma=sigma_hat)

        if self.signal_type == "grey":
            y_img = y.reshape(self.im_size)
            output = convolve(y_img, h)
            output_vec = output.reshape(-1, 1)
        else:
            y_img = y.reshape((self.im_size[0], self.im_size[1], 3))
            for i in range(3):
                y_img[..., i] = convolve(y_img[..., i], h)

            output_vec = y_img.reshape(-1, 3)

        return output_vec
    
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h