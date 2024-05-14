import time

import torch
import torch.nn as nn

from deeplearning.trainer.measure import AverageMeter

class ClassifierTrainer(nn.Module):
    def __init__(self, config, model):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.config = config

    def forward(self, contents):
        """
        Algorithm of training a classifier 
        """
        target = contents[1].to(self.config.device)
        input = contents[0].to(self.config.device)

        # Compute output
        output = self.model(input)
        loss = self.criterion(output, target).mean()

        return loss
    

def test_classifier(test_loader, network, criterion, config, logger, record):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy_meter = AverageMeter('Accs', ':6.2f')

    # Switch to evaluate mode
    network.eval()
    network.no_grad = True

    end = time.time()
    for i, data in enumerate(test_loader):
        input, target = data[0], data[1]

        target = target.to(config.device)
        input = input.to(config.device)

        # Compute output
        with torch.no_grad():
            output = network(input)

            loss = criterion(output, target).mean()

        # Measure accuracy and record loss
        acc = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        accuracy_meter.update(acc.item(), input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    logger.info('Test acc: * Acc {accuracy.avg:.3f}'.format(accuracy=accuracy_meter))
    
    record["test_loss"].append(losses.avg)
    record["test_acc"].append(accuracy_meter.avg)

    network.no_grad = False

    return accuracy_meter.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res