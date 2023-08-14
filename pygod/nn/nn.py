# -*- coding: utf-8 -*-
"""Personalized neural network layers used by the model."""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import torch
import torch.nn as nn
    

class MLP_GAD_NR(torch.nn.Module):
    r"""
    The personalized MLP module used by GAD_NR
    Source: https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR.ipynb
    
    Parameters
    ----------
    in_dim : int
        Input dimension of the embedding.
    hid_dim :  int
        Hidden dimension of model.
    out_dim : int
        Output dimension.
    num_layers : int
        Number of layers in the decoder.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``. 
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 act=torch.nn.functional.relu):
        super(MLP_GAD_NR, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.act = act

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(in_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.linears.append(nn.Linear(hid_dim, out_dim))

            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hid_dim)))

    def forward(self, x):
        r"""
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding.

        Returns
        -------
        h : torch.Tensor
            Transformed embeddings.
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                
                if len(h.shape) > 2:
                    h = torch.transpose(h, 0, 1)
                    h = torch.transpose(h, 1, 2)
                    
                h = self.batch_norms[layer](h)
                
                if len(h.shape) > 2:
                    h = torch.transpose(h, 1, 2)
                    h = torch.transpose(h, 0, 1)

                h = self.act(h)
                h = self.linears[self.num_layers - 1](h)
                
            return h 


class MLP_generator(nn.Module):
    r"""
    The personalized MLP module used by GAD_NR
    Source: https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR.ipynb
    
    Parameters
    ----------
    in_dim : int
        Input dimension of the embedding.
    out_dim : int
        Output dimension.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``. 
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 act=torch.nn.functional.relu):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.linear3 = nn.Linear(out_dim, out_dim)
        self.linear4 = nn.Linear(out_dim, out_dim)
        self.act = act

    def forward(self, emb):
        r"""
        Forward computation.

        Parameters
        ----------
        emb : torch.Tensor
            Input embedding.

        Returns
        -------
        neighbor_emb : torch.Tensor
            Output neighbor embedding.
        """
        neighbor_emb = self.act(self.linear(emb))
        neighbor_emb = self.act(self.linear2(neighbor_emb))
        neighbor_emb = self.act(self.linear3(neighbor_emb))
        neighbor_emb = self.linear4(neighbor_emb)
        return neighbor_emb


class FNN_GAD_NR(nn.Module):
    r"""
    The personalized FNN module used by GAD_NR
    Source: https://github.com/Graph-COM/GAD-NR/blob/master/GAD-NR.ipynb
    
    Parameters
    ----------
    in_dim : int
        Input dimension of the embedding.
    hid_dim :  int
        Hidden dimension of model.
    out_dim : int
        Output dimension.
    num_layers : int
        Number of layers in the decoder.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    """
    def __init__(self,
                 in_dim,
                 hid_dim, 
                 out_dim,
                 num_layers,
                 act=torch.nn.functional.relu):
        super(FNN_GAD_NR, self).__init__()
        self.act = act
        self.linear1 = MLP_GAD_NR(num_layers, in_dim, hid_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
    
    def forward(self, emb):
        r"""
        Forward computation.

        Parameters
        ----------
        emb : torch.Tensor
            Input embedding.

        Returns
        -------
        x_ : torch.Tensor
            Output embedding.
        """
        emb = self.linear1(emb)
        emb = self.linear2(self.act(emb))
        x_ = self.act(emb)
        return x_
