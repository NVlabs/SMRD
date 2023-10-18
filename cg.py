import torch
from torch import nn

class ConjugateGradient(nn.Module):
    """
    Implementation of conjugate gradient algorithm based on: 
    https://github.com/utcsilab/deepinpy/blob/master/deepinpy/opt/conjgrad.py
    """

    def __init__(self, A, num_iter, dbprint=False):
        super(ConjugateGradient, self).__init__()

        self.A = A
        self.num_iter = num_iter
        self.dbprint = dbprint

    def zdot(self, x1, x2):
        """
        Complex dot product between tensors x1 and x2.
        """
        return torch.sum(x1.conj() * x2)

    def zdot_single(self, x):
        """
        Complex dot product between tensor x and itself
        """
        return self.zdot(x, x).real

    def _update(self, iter):
        def update_fn(x, p, r, rsold):
            Ap = self.A(p)
            pAp = self.zdot(p, Ap)
            alpha = (rsold / pAp)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = self.zdot_single(r)
            beta = (rsnew / rsold)
            rsold = rsnew
            p = beta * p + r

            # print residual
            if self.dbprint:
                print(f'CG Iteration {iter}: {rsnew}')

            return x, p, r, rsold

        return update_fn

    def forward(self, x, y):
        # Compute residual
        r = y - self.A(x)
        rsold = self.zdot_single(r)
        p = r

        for i in range(self.num_iter):
            x, p, r, rsold = self._update(i)(x, p, r, rsold)

        return x
    
    def reverse(self, x):
        out = (1/self.lamb) * (self.A(x) + self.lamb*x - self.Aty)
        return out
