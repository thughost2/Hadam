from torch.optim import *
import torch
import math
import numpy as np
 

class Hadam(Optimizer):
    """Implements Hadam algorithm.
    It combines HMC method with Adam algorithm
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta2 (float, optional): coefficients used for computing
            running averages of gradient square (default: 0.999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad type of variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
        gamma (float, optional): temperature parameter for HMC method
        fraction (float, optional): fraction parameter for HMC method
        eta (float, optional): step size parameter for momentum accumulation
        bias_correction (boolean, optional): whether to perform bias correction as in Adam algorithm
    """

    def __init__(self, params, lr=1e-3, beta2 = 0.999, eps=1e-8,
                 weight_decay=0, amsgrad=False, gamma = 10000, fraction = 0.5, eta = 0.1, bias_correction = False):
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter beta2: {}".format(beta2))
        defaults = dict(lr=lr, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, gamma = gamma, fraction = fraction, eta = eta, bias_correction = bias_correction)
        super(Hadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsgrad = group['amsgrad']
                lr = group['lr']
                eta = group['eta']
                fraction = group['fraction']
                bias_correction = group['bias_correction']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['p_momentum'] = torch.zeros_like(grad)
                    
                p_momentum = state['p_momentum']
                exp_avg_sq = state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta2 = group['beta2']
                gamma = group['gamma']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                                    
                if bias_correction == True:
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = lr * math.sqrt(bias_correction2) 
                else:
                    step_size = lr
                                
                # add SGLD Brownian motion noise
                noise = torch.zeros_like(grad).normal_()
                grad.add_(math.sqrt(2*step_size/gamma/state['step']), noise)
   
                
                    
                p_momentum.mul_(1 - fraction*eta).add_(- eta, grad)    
                    
                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(step_size, p_momentum, denom)
                

        return loss
    
    
    
    
    
    
    
    

 