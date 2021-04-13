import copy
import math

import torch
from torch.optim.optimizer import Optimizer


class Adam16(Optimizer):
    """
    https://gist.github.com/ajbrock/075c0ca4036dc4d8581990a6e76e07a3
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

        self.fp32_param_groups = [p.cuda() for p in params]
        if not isinstance(self.fp32_param_groups[0], dict):
            self.fp32_param_groups = [{"params": self.fp32_param_groups}]

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, fp32_group in zip(self.param_groups, self.fp32_param_groups):
            for p, fp32_p in zip(group["params"], fp32_group["params"]):
                a = copy.deepcopy(fp32_p)
                fp32_p = p.data.float()
                if p.grad is None:
                    continue

                grad = p.grad.data.float()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], fp32_p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # print(type(fp32_p))
                fp32_p.addcdiv_(-step_size, exp_avg, denom)
                if a.dtype == torch.float16:
                    p.data = fp32_p.half()

        return loss
