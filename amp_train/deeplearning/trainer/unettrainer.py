import torch
import torch.nn as nn

from deeplearning.trainer.measure import AverageMeter
from deeplearning.loss import PerceptualLoss, StyleLoss

class UnetTrainer(nn.Module):
    def __init__(self, config, model):
        super().__init__()

        self.model = model
        self.config = config

        # self.criterion = nn.MSELoss()
        # self.dist_loss = nn.L1Loss()
        self.dist_loss = nn.MSELoss()
        # # adv_loss = AdvLoss()
        self.percep_loss = PerceptualLoss()
        self.style_loss = StyleLoss()


    def forward(self, x, noisy_x, std):
        """
        Algorithm of training adaptive regressor/denoiser
        """
        num_samples = noisy_x.shape[0]

        # prior distribution for sigma^2
        # 1. inverse gamma distribution InvGamma(shape, scale)
        # shape, scale = self.config.shape, self.config.scale

        # # Generating sample data
        # s2_numpy = invgamma.rvs(shape, scale=scale, size=num_samples)
        # s2 = torch.from_numpy(s2_numpy).to(x)
        # s_ = torch.zeros_like(s2).to(torch.int)

        # noisy_data = torch.clone(x)
        # noise = torch.zeros_like(x)
        # for i in range(num_samples):
        #     # s = torch.sqrt(s2[i])
        #     s = std[]
        #     noise[i] = torch.randn_like(x[i]) * std / self.config.denominator
        #     noisy_data[i] += noise[i]
        #     index_ = torch.round(s)
        #     index_ = index_.to(torch.int) if index_ < self.upperbound else self.upperbound - 1
            # s_[i] = index_

        # print(s_)

        pred = self.model(noisy_x, std)
        ref = x
        # ref = noise

        # denoised = self.model(noisy_data, s_)
        # loss = self.criterion(denoised, x)

        dist_loss = self.dist_loss(pred, ref)
        percep_loss = self.percep_loss(pred, ref)
        style_loss = self.style_loss(pred, ref)

        # percep_loss, style_loss = 0., 0.

        loss = self.config.l1 * dist_loss
        loss += self.config.percep * percep_loss
        loss += self.config.style * style_loss
        
        # pred = self.model(noisy_data, s_)
        # loss = self.criterion(pred, noise)

        return  pred, loss, (dist_loss, percep_loss, style_loss)
    