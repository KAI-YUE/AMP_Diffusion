import time

# PyTorch libraries
import torch

# My libraries
from config import load_config
from config.utils import *
from amp.utils import damp_snr, damps_snr_rgb, sample_save
from amp.signalprocess import generate_signal
from deeplearning.datasets import fetch_dataloader, fetch_dataset


def damp(config, dataset, logger):
    record = init_record()

    dst_train, dst_test = dataset.dst_train, dataset.dst_test
    train_loader, test_loader = fetch_dataloader(config, dst_train, dst_test)

    ys, As, raw_signal, sigmas, A_perturbs = generate_signal(config, train_loader)
    
    # num_realizations = len(config.snr_db)
    num_realizations = len(config.Aperturb_snr)
    for i in range(num_realizations):
        start = time.time()
        
        # a perturbed A
        x_hat, mmse, nmse, psnr = damps_snr_rgb(config, ys[i], raw_signal, A_perturbs[i], logger, sigma_init=sigmas[i])
        end = time.time()
        logger.info("Time elapsed: {:.3}s".format((end-start)))

        # output the metrics here 
        logger.info("MSE: {:.3} \t NMSE: {:.3} \t PSNR {:.3} dB".format(mmse, nmse, psnr))
        record["mmse"].append(mmse)
        record["psnr"].append(psnr)


    # visualize the results
    if not config.output2folder:
        sample_save(config, raw_signal, "raw", pixel_scaling=config.pixel_scaling)
        sample_save(config, x_hat, config.denoiser, pixel_scaling=config.pixel_scaling)

    # save the record
    save_record(record, config.output_dir)

def main():

    config = load_config()
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir, config.seed, attach=True)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    torch.random.manual_seed(config.seed)
    np.random.seed(config.seed)

    # start = time.time()
    dataset = fetch_dataset(config)
    damp(config, dataset, logger)
    # end = time.time()

    # logger.info("{:.3} mins has elapsed".format((end-start)/60))
    logger.handlers.clear()


if __name__ == "__main__":
    main()

