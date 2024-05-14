import time

# PyTorch libraries
import torch

# My libraries
from config import load_config
from config.utils import *
from deeplearning.utils import *
from deeplearning.datasets import *
from deeplearning.trainer import test_regressor

def train_ampnet(config, dataset, logger):
    record = init_record()

    dst_train, dst_test = dataset.dst_train, dataset.dst_test
    model, criterion, optimizer, scheduler, start_epoch = init_all(config, dataset, logger)
    
    train_loader, test_loader = fetch_noisydataloader(config, dst_train, dst_test)

    lowest_err, best_epoch = np.inf, 0

    for epoch in range(start_epoch, config.epochs):
        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, config, logger, record)
        
        # evaluate 
        if config.test_interval > 0 and (epoch + 1) % config.test_interval == 0:
            metric = test_regressor(test_loader, model, epoch, config, logger, record)

            # remember best prec@1 and save checkpoint
            is_best = metric < lowest_err

            if is_best:
                lowest_err = metric
                best_epoch = epoch + 1
                save_checkpoint({"epoch": epoch + 1,
                                "state_dict": model.state_dict(),
                                "opt_dict": optimizer.state_dict(),
                                "best_epoch": best_epoch},
                                config.output_dir + "/checkpoint_epoch{:d}.pth".format(epoch))
                                # config.output_dir + "/checkpoint_epoch.pth")

    # save the record
    save_record(record, config.output_dir)
        
def main():
    # load the config file, logger, and initialize the output folder
    config = load_config()
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir, config.seed, attach=True)

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True

    torch.random.manual_seed(config.seed)
    np.random.seed(config.seed)

    start = time.time()
    dataset = fetch_dataset(config)
    train_ampnet(config, dataset, logger)
    end = time.time()

    logger.info("{:.3} mins has elapsed".format((end-start)/60))
    logger.handlers.clear()

if __name__ == "__main__":
    main()

