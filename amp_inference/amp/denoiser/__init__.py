from .gaussian import GaussianDenoiser
from .ccunetdenoiser import CCUnetDenoiser

denoiser_registry = {
    'gaussian': GaussianDenoiser,
    'ccunet': CCUnetDenoiser,
}