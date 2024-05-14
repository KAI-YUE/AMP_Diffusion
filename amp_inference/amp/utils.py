import os
import cv2
import numpy as np
import copy
import random

from amp.denoiser import denoiser_registry

def damp_snr(config, y, raw, Amat, logger=None, sigma_init=1., **kwargs):

    if config.debug:
        return damp_debug(config, y, raw, Amat, logger)
    
    if config.signal_type == "rgb":
        return damp_snr_rgb(config, y, raw, Amat, logger, sigma_init)

    denoiser = denoiser_registry[config.denoiser](config)

    A = lambda x: Amat @ x.reshape(-1, 1)
    At = lambda z: Amat.T @ z.reshape(-1, 1)

    # denoise(noisy, sigma_hat)
    N = config.im_size[0] * config.im_size[1]
    M = len(y)

    z_t = y.reshape(-1, 1)
    x_t = np.zeros(N)

    PSNR = np.zeros(config.amp_iters)
    mmse = np.zeros(config.amp_iters)

    for i in range(config.amp_iters):
        if logger is not None:
            logger.info("-------------- AMP iteration {} --------------".format(i))

        pseudo_data = At(z_t) + x_t.reshape(-1, 1)
        sigma_hat = np.sqrt(1/M * np.sum(z_t**2))
        x_t = denoiser(pseudo_data, sigma_hat)

        sample_save(config, x_t, "iter_{:d}".format(i), pixel_scaling=config.pixel_scaling)

        mmse[i] = MSE_func(x_t, raw, config.pixel_scaling)
        PSNR[i] = PSNR_func(mmse[i])
        
        eta = np.random.randn(N, 1)
        epsilon = np.max(pseudo_data)/1000 + np.finfo(float).eps
        
        # a little wonder along the data
        pseudo_denoised = denoiser(pseudo_data + epsilon*eta, sigma_hat)
        div = eta.T @ (pseudo_denoised - x_t) / epsilon
        z_t = y.reshape(-1, 1) - A(x_t) + (1/M) * z_t * div[0, 0]

        logger.info("Sigma_hat {:.2f} \t MMSE: {:.2f} \t PSNR {:.2f} dB".format(sigma_hat, mmse[i], PSNR[i]))
        logger.info("-"*80)

    x_hat = x_t.reshape((config.im_size[0], config.im_size[1]))
    mmse_final = mmse[-1]
    psnr_final = PSNR[-1]

    return x_hat, mmse_final, psnr_final


def damp_snr_rgb(config, y, raw, Amat, logger=None, sigma_init=1., **kwargs):

    if config.debug:
        return damp_debug(config, y, raw, Amat, logger)

    denoiser = denoiser_registry[config.denoiser](config)

    A = lambda x: Amat @ x.reshape(-1, config.channel)
    At = lambda z: Amat.T @ z.reshape(-1, config.channel)

    # denoise(noisy, sigma_hat)
    N = config.im_size[0] * config.im_size[1]
    M = y.shape[0]

    # z_ts = np.hstack([y[0].reshape(-1, 1), y[1].reshape(-1, 1), y[2].reshape(-1, 1)])
    z_t = copy.deepcopy(y) 
    x_t = np.zeros((N, config.channel))
    pseudo_datas = np.zeros((N, config.channel))
    sigma_hats = sigma_init

    PSNR = np.zeros(config.amp_iters)
    mmse = np.zeros(config.amp_iters)

    for i in range(config.amp_iters):
        if logger is not None:
            logger.info("-------------- AMP iteration {} --------------".format(i))

        # process three channels
        # for j in range(config.channel):
        #     z_t = z_ts[:, j].reshape(-1, 1)
        #     x_t = x_ts[:, j].reshape(-1, 1)
        #     pseudo_datas[:, j] = At(z_t) + x_t
        #     sigma_hats[j] = np.sqrt(1/M * np.sum(z_t**2))

        pseudo_datas = At(z_t) + x_t
        x_t = denoiser(pseudo_datas, sigma_hats)

        sample_save(config, x_t, "iter_{:d}".format(i), pixel_scaling=config.pixel_scaling)

        mmse[i] = MSE_func(x_t, raw, config.pixel_scaling)
        PSNR[i] = PSNR_func(mmse[i])
        
        eta = np.random.randn(N, config.channel)
        epsilon = np.max(pseudo_datas)/100
        
        # a little wonder along the data
        # for j, (z_t, x_t) in enumerate(zip(z_ts, x_ts)):
        pseudo_denoised = denoiser(pseudo_datas + epsilon*eta, sigma_hats)
        div = eta.T @ (pseudo_denoised - x_t) / epsilon
        z_t = y - A(x_t) + (1/(config.channel*M)) * z_t * div.mean()

        sigma_hats = np.sqrt(1/(config.channel*M) * np.sum(z_t**2))

        logger.info("Sigma_hat {:.2f} \t MMSE: {:.2f} \t PSNR {:.2f} dB".format(sigma_hats, mmse[i], PSNR[i]))
        logger.info("-"*80)

    x_hat = np.zeros((config.im_size[0], config.im_size[1], config.channel))
    for j in range(config.channel):
        x_hat[:, :, j] = x_t[:, j].reshape((config.im_size[0], config.im_size[1]))

    mmse_final = mmse[-1]
    psnr_final = PSNR[-1]

    return x_hat, mmse_final, psnr_final


def damp_debug(config, y, raw, Amat, logger=None, **kwargs):
    """
    This is for debug purpose as a sanity check. 
    """

    denoiser = denoiser_registry[config.denoiser](config)
    # x_hat = denoiser(raw, 1)
    x_hat = denoiser(y, 1)

    mmse_final = MSE_func(x_hat, raw, config.pixel_scaling)
    psnr_final = PSNR_func(mmse_final)

    return x_hat, mmse_final, psnr_final


def damps_snr_rgb(config, y, raw, Amat, logger=None, sigma_init=1.):
    """
    instead of using 1 denoiser, we use a list of denoisers this time 
    """

    if config.debug:
        return damp_debug(config, y, raw, Amat, logger)

    denoisers = []
    for denoiser_name in config.denoisers:
        denoisers.append(denoiser_registry[denoiser_name](config))

    A = lambda x: Amat @ x.reshape(-1, config.channel)
    At = lambda z: Amat.T @ z.reshape(-1, config.channel)

    # denoise(noisy, sigma_hat)
    N = config.im_size[0] * config.im_size[1]
    M = y.shape[0]

    z_t = copy.deepcopy(y.reshape(-1, config.channel))
    x_t = np.zeros((N, config.channel))
    pseudo_data = np.zeros((N, config.channel))
    sigma_hat = sigma_init

    PSNR = np.zeros(config.amp_iters)
    mmse = np.zeros(config.amp_iters)
    nmse = np.zeros(config.amp_iters)

    for i in range(config.amp_iters):
        if logger is not None:
            logger.info("-------------- AMP iteration {} --------------".format(i))

        est_sigma_per_iter = np.zeros(len(denoisers))
        x_t_per_iter, z_t_per_iter = [], []

        pseudo_data = At(z_t) + x_t
        sigma_hat = np.sqrt(1/(config.channel*M) * np.sum(z_t**2))
        
        sample_save(config, pseudo_data, "pseudo_{:d}".format(i), pixel_scaling=config.pixel_scaling)

        eta = np.random.randn(N, config.channel)
        epsilon = np.max(pseudo_data)/100

        for j, denoiser in enumerate(denoisers):
            config.denoiser = config.denoisers[j]
            x_t = denoiser(pseudo_data, sigma_hat)
            x_t_per_iter.append(x_t)
            
            # a little wonder along the data
            pseudo_denoised = denoiser(pseudo_data + epsilon*eta, sigma_hat)
            div = eta.T @ (pseudo_denoised - x_t) / epsilon
            z_t = y.reshape(-1, config.channel) - A(x_t) + (1/(config.channel*M)) * z_t * div.mean()

            est_sigma =  np.sqrt(1/(config.channel*M) * np.sum(z_t**2))
            est_sigma_per_iter[j] = est_sigma
            z_t_per_iter.append(z_t)

        # pick the best denoiser
        best_idx = np.argmin(est_sigma_per_iter)
        print(est_sigma_per_iter)
        x_t = x_t_per_iter[best_idx]
        z_t = z_t_per_iter[best_idx]

        # Gaussian denoiser, we apply nn again?
        # if config.denoisers[best_idx] == "gaussian":
        #     x_t = denoisers[-1](pseudo_data, est_sigma_per_iter[best_idx])
        
        sample_save(config, x_t, "iter_{:d}".format(i), pixel_scaling=config.pixel_scaling)
        
        if config.output2folder:
            output_to_folder(config, pseudo_data, raw, sigma=sigma_hat, pixel_scaling=config.pixel_scaling, folder=config.synthetic_folder)

        mmse[i] = MSE_func(x_t, raw, config.pixel_scaling)
        PSNR[i] = PSNR_func(mmse[i])
        nmse[i] = NMSE_func(x_t, raw, config.pixel_scaling)

        logger.info("Selected denoiser: {}".format(config.denoisers[best_idx]))
        logger.info("Sigma_hat {:.2f} \t MMSE: {:.2f} \t PSNR {:.2f} dB".format(sigma_hat, mmse[i], PSNR[i]))
        logger.info("-"*80)

    x_hat = x_t
    mmse_final = mmse[-1]
    psnr_final = PSNR[-1]

    return x_hat, mmse_final, nmse[-1], psnr_final

def MSE_func(original, reconstructed, pixel_scaling=1):
    """
    Calculate the Mean Squared Error between the original and reconstructed image.
    
    :param original: numpy array of the original image.
    :param reconstructed: numpy array of the reconstructed image.
    :return: Mean Squared Error.
    """
    mse = np.mean((pixel_scaling*original - pixel_scaling*reconstructed) ** 2)
    return mse

def NMSE_func(original, reconstructed, pixel_scaling=1):
    """
    Calculate the Mean Squared Error between the original and reconstructed image.
    
    :param original: numpy array of the original image.
    :param reconstructed: numpy array of the reconstructed image.
    :return: Mean Squared Error.
    """
    mse = np.sum((pixel_scaling*original - pixel_scaling*reconstructed) ** 2)/np.sum((pixel_scaling*original) ** 2)
    return mse


def PSNR_func(mse, pixel_bound=255):
    """
    Calculate the Peak Signal-to-Noise Ratio between the original and reconstructed image.
    
    :param original: numpy array of the original image.
    :param reconstructed: numpy array of the reconstructed image.
    :param pixel_bound: The maximum pixel value of the image (255 for 8-bit images).
    :return: Peak Signal-to-Noise Ratio in decibels.
    """
    if mse == 0:
        # Means no difference between original and reconstructed image.
        return float('inf')
    psnr = 20 * np.log10(pixel_bound / np.sqrt(mse))
    return psnr


def sample_save(config, x_hat, info="", pixel_scaling=255):
    if config.signal_type == "grey":
        img = x_hat.reshape((config.im_size[0], config.im_size[1]))
    else:
        img = np.zeros((config.im_size[0], config.im_size[1], 3))
        for j in range(config.channel):
            img [:, :, j] = x_hat[..., j].reshape((config.im_size[0], config.im_size[1]))
        img = img[..., ::-1]

    img_uint8 = np.clip(pixel_scaling*img, 0, 255).astype(np.uint8)
    
    if not os.path.exists(os.path.join(config.output_dir, "img")):
        os.makedirs(os.path.join(config.output_dir, "img"))
    
    cv2.imwrite(os.path.join(config.output_dir, "img", info+"output.png"), img_uint8)


def output_to_folder(config, x_hat, raw, sigma, pixel_scaling=255, folder="synthetic"):
    if config.signal_type == "grey":
        img = x_hat.reshape((config.im_size[0], config.im_size[1]))
        raw_img = raw.reshape((config.im_size[0], config.im_size[1]))
    else:
        img = np.zeros((config.im_size[0], config.im_size[1], 3))
        raw_img = np.zeros((config.im_size[0], config.im_size[1], 3))
        for j in range(config.channel):
            img [:, :, j] = x_hat[..., j].reshape((config.im_size[0], config.im_size[1]))
            raw_img [:, :, j] = raw[..., j].reshape((config.im_size[0], config.im_size[1]))
        
        img = img[..., ::-1]
        raw_img = raw_img[..., ::-1]

    img_uint8 = np.clip(pixel_scaling*img, 0, 255).astype(np.uint8)
    raw_img_uint8 = np.clip(pixel_scaling*raw_img, 0, 255).astype(np.uint8)

    if not os.path.exists(os.path.join(config.output_dir, "img")):
        os.makedirs(os.path.join(config.output_dir, "img"))
    
    # Random letter suffix
    # Generate a alphabeta firstly
    CodeLen = 10
    alphabeta = []
    for letter in range(65,92):
        alphabeta.append(chr(letter)) 

    for letter in range(97,124):
        alphabeta.append(chr(letter))

    random_code = random.sample(alphabeta,CodeLen)

    if not os.path.exists(os.path.join(config.synthetic_folder, "{:d}".format(int(sigma)))):
        os.makedirs(os.path.join(config.synthetic_folder, "{:d}".format(int(sigma))))
    cv2.imwrite(os.path.join(config.synthetic_folder, "{:d}".format(int(sigma)), "".join(random_code) + ".jpg"), img_uint8)

    # save the raw image
    if not os.path.exists(os.path.join(config.groundtruth_folder, "{:d}".format(int(sigma)))):
        os.makedirs(os.path.join(config.groundtruth_folder, "{:d}".format(int(sigma))))
    cv2.imwrite(os.path.join(config.groundtruth_folder, "{:d}".format(int(sigma)), "".join(random_code) + ".jpg"), raw_img_uint8)