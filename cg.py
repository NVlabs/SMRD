# MIT License
"""
MIT License

Copyright (c) 2021 The University of Texas Computational Sensing and Imaging Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
------------------------------------------------------------------------------
Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This work is made available under the Nvidia Source Code License.
To view a copy of this license, visit
https://github.com/batuozt/SMRD/blob/master/LICENSE.md

Written by Batu Ozturkler
------------------------------------------------------------------------------
"""

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
