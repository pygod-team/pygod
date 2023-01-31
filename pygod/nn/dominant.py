import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCN, GraphSAGE

from .functional import double_mse_loss


class DOMINANTBase(nn.Module):
    """
       DOMINANT (Deep Anomaly Detection on Attributed Networks) is an
       anomaly detector consisting of a shared graph convolutional
       encoder, a structure reconstruction decoder, and an attribute
       reconstruction decoder. The reconstruction mean square error of the
       decoders are defined as structure anomaly score and attribute
       anomaly score, respectively.

       See :cite:`ding2019deep` for details.

       Parameters
       ----------
       TODO: update the docstring here
       hid_dim :  int, optional
           Hidden dimension of model. Default: ``0``.
       num_layers : int, optional
           Total number of layers in model. A half (floor) of the layers
           are for the encoder, the other half (ceil) of the layers are
           for decoders. Default: ``4``.
       dropout : float, optional
           Dropout rate. Default: ``0.``.
       weight_decay : float, optional
           Weight decay (L2 penalty). Default: ``0.``.
       act : callable activation function or None, optional
           Activation function if not None.
           Default: ``torch.nn.functional.relu``.
       alpha : float, optional
           Loss balance weight for attribute and structure. ``None`` for
           balancing by standard deviation. Default: ``None``.
       contamination : float, optional
           Valid in (0., 0.5). The proportion of outliers in the data set.
           Used when fitting to define the threshold on the decision
           function. Default: ``0.1``.
       lr : float, optional
           Learning rate. Default: ``0.004``.
       epoch : int, optional
           Maximum number of training epoch. Default: ``5``.
       gpu : int
           GPU Index, -1 for using CPU. Default: ``0``.
       batch_size : int, optional
           Minibatch size, 0 for full batch training. Default: ``0``.
       num_neigh : int, optional
           Number of neighbors in sampling, -1 for all neighbors.
           Default: ``-1``.
       verbose : bool
           Verbosity mode. Turn on to print out log information.
           Default: ``False``.
       """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act,
                 sigmoid_s=False,
                 scalable=False,
                 **kwargs):
        super(DOMINANTBase, self).__init__()

        self.sigmoid_s = sigmoid_s
        self.scalable = scalable

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)


        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.attr_decoder = GCN(in_channels=hid_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=in_dim,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=decoder_layers - 1,
                                  out_channels=in_dim,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.loss_func = double_mse_loss

    def forward(self, x, edge_index):
        # encode
        h = self.shared_encoder(x, edge_index)
        # decode feature matrix
        x_ = self.attr_decoder(h, edge_index)

        if not self.scalable:
            # decode adjacency matrix, which is used by the vanilla DOMINANT
            h_ = self.struct_decoder(h, edge_index)
            s_ = h_ @ h_.T
            # return reconstructed matrices
            if self.sigmoid_s:
                s_ = torch.sigmoid(s_)
        else:
            # decode the center node structural embedding, which is more scalable
            s_ = self.struct_decoder(h, edge_index)

        return x_, s_
