import numpy as np

from .imggenerator import ImageGenerator, greytensor_2_npvec, rgbtensor_2_npvec, npvec2tensor


signal_registry = {
    'img': ImageGenerator,
}

def generate_signal(config, dataloader, **kwargs):
    # generate projection matrix

    N = config.N
    M = round(config.subrate * N)

    # generate signal
    signal_generator = signal_registry[config.signal_source](dataloader=dataloader, config=config)
    raw_signal = signal_generator.generate()

    # num_realizations = len(config.snr_db)
    num_realizations = len(config.Aperturb_snr)
    snr_linear = (10**(np.array(config.snr_db)/10)).flatten()

    ys = np.zeros((num_realizations, M)) if config.signal_type == "grey" else np.zeros((num_realizations, M, 3))
    As = np.zeros((num_realizations, M, N))
    A_perturbs = np.zeros((num_realizations, M, N))
    sigmas = np.zeros(num_realizations)

    ch = 1 if config.signal_type == "grey" else 3

    # a list of noisy signals
    for i in range(num_realizations):
        A = np.random.randn(M, N)/np.sqrt(N);   # random Gaussian operator
        # generate identity matrix for debug
        # A = np.eye(N)

        z = A @ raw_signal
        # nvar = np.linalg.norm(z)**2/(len(z) * snr_linear[i])
        nvar = np.linalg.norm(z)**2/(len(z) * snr_linear[0])

        sigmas[i] = np.sqrt(nvar)
        noise = sigmas[i] * np.random.randn(M, ch) # zero mean additive white Gaussian noise    
        y = z + noise

        ys[i, :] = y.flatten() if config.signal_type == "grey" else y
        As[i, :, :] = A

        # now perturb the A a bit 
        Aperturb_snr_linear = 10**(np.asarray(config.Aperturb_snr)/10)
        A_perturbs[i, :, :] = A + np.sqrt(1/(N*Aperturb_snr_linear[i])) * np.random.randn(M, N)

    return ys, As, raw_signal, sigmas, A_perturbs



