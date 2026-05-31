# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
from torch import nn


class SAM_SPS(torch.optim.Optimizer):
    r"""
    Implements SAM / USAM equipped with the Stochastic Polyak Scheduler.

    e^t = x^t + \rho * (1 - \lambda + \lambda / \|\nabla f_i(x^t)\|) * \nabla f_i(x^t)
    \gamma_t = min{ (f_i(e^t) - \ell^* - <\nabla f_i(e^t), e^t - x^t>) / \|\nabla f_i(e^t)\|^2 , \gamma_b }
    x^{t+1} = x^t - \gamma_t * \nabla f_i(e^t)

    With \lambda = 0 this is USAM-SPS; with \lambda = 1 this is SAM-SPS.
    """

    def __init__(self,
                 params: Iterable[nn.parameter.Parameter],
                 weight_decay: float = 5e-4,
                 rho: float = 0.1,
                 lambd: float = 1.0,
                 f_star: float = 0.0,
                 gamma_b: float = 1.0):
        """
        Arguments:
            params (iterable):
                Iterable of parameters to optimize or dicts defining parameter groups.
            weight_decay (float):
                L2 weight-decay coefficient applied in the final SGD-style update.
            rho (float):
                The sharpness radius \rho.
            lambd (float):
                Interpolates between USAM (\lambda = 0) and SAM (\lambda = 1).
            f_star (float):
                Lower bound \ell^* on the (mini-batch) loss. Typically 0 for non-negative losses.
            gamma_b (float):
                Upper bound \gamma_b on the Stochastic Polyak Scheduler step size.
        """
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")

        # lr kept only for logging compatibility; overwritten each step by \gamma_t and used in SGD update rule
        defaults = dict(lr=gamma_b, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.rho = rho
        self.lambd = lambd
        self.f_star = f_star
        self.gamma_b = gamma_b

        self.grad_norm = torch.tensor(0.0)

    @torch.no_grad()
    def _sgd_update(self, group):
        lr = group['lr']
        wd = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad

            if wd != 0.0:
                d_p = d_p.add(p, alpha=wd)

            p.add_(d_p, alpha=-lr)

    def step(self, closure):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure : callable
            A closure that re-evaluates the model and returns the (mini-batch) loss.
            Required by SAM-style optimizers because the gradient must be computed
            twice (at x^t and at the perturbed point e^t).

        Returns
        -------
        (Stochastic) Loss function value at the perturbed point e^t.
        """
        eps = 1e-12
        if closure is None:
            raise ValueError("SAM_SPS.step() requires a closure")

        closure = torch.enable_grad()(closure)

        # Pass 1: grads at x^t
        self.zero_grad(set_to_none=True)
        loss_x = closure()

        grad_norm_sq_x = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad_x = p.grad.detach().to(torch.float32)
                    grad_norm_sq_x = grad_x.pow(2).sum() if grad_norm_sq_x is None else grad_norm_sq_x + grad_x.pow(2).sum()
        self.grad_norm = grad_norm_sq_x.sqrt().clamp(min=eps)

        # Build e^t and stash e^t - x^t per parameter
        scale = self.rho * ((1 - self.lambd) + self.lambd / self.grad_norm)
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    rest = (p.grad * scale).to(dtype=p.dtype)
                    p.add_(rest)
                    self.state[p]['e^t-x^t'] = rest

        # Pass 2: grads at e^t
        self.zero_grad(set_to_none=True)
        loss_e = closure()

        # Compute \gamma_t via the Stochastic Polyak Scheduler
        dot = None
        grad_norm_sq_e = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                rest = self.state.get(p, {}).get('e^t-x^t', None)
                if rest is None:
                    continue
                rest = rest.detach().to(torch.float32)
                grad_e = p.grad.detach().to(torch.float32)
                dot = torch.sum(grad_e * rest) if dot is None else dot + torch.sum(grad_e * rest)
                grad_norm_sq_e = grad_e.pow(2).sum() if grad_norm_sq_e is None else grad_norm_sq_e + grad_e.pow(2).sum()

        num = loss_e.detach().to(torch.float32) - self.f_star - dot
        denom = grad_norm_sq_e.clamp(min=eps)
        gamma = max(0.0, min((num / denom).item(), self.gamma_b))

        for group in self.param_groups:
            group['lr'] = gamma

        # Restore p to x^t (back from e^t)
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    rest = self.state.get(p, {}).get('e^t-x^t', None)
                    if rest is not None:
                        p.sub_(rest)
                        self.state[p].pop('e^t-x^t', None)

        # SGD update using \nabla f_i(e^t)
        with torch.no_grad():
            for group in self.param_groups:
                self._sgd_update(group)

        return loss_e
