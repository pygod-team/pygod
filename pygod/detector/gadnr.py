# -*- coding: utf-8 -*-
"""GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction (GADNR)
   The code is partially from the original implementation in 
   https://github.com/Graph-COM/GAD-NR"""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GIN
from torch_geometric import compile

from . import DeepDetector
from ..nn import GADNRBase
from ..utils import logger


class GADNR(DeepDetector):
    """
    TODO finish the docstring

    real_loss : bool, optional
    Whether using the original loss proposed in the paper as the
    decision score, if not, using the proposed adaptive decision score.
    Default: ``False``.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=1,
                 deg_dec_layers=4,
                 fea_dec_layers=3,
                 backbone=GIN,
                 sample_size=2,
                 sample_time=3,
                 neigh_loss='KL',
                 lambda_loss1=0.01,
                 lambda_loss2=0.1,
                 lambda_loss3=0.8,
                 real_loss=False,
                 lr=0.01,
                 epoch=500,
                 dropout=0.,
                 weight_decay=0.0003,
                 act=F.relu,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 contamination=0.1,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(GADNR, self).__init__(hid_dim=hid_dim,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    backbone=backbone,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    save_emb=save_emb,
                                    compile_model=compile_model,
                                    **kwargs)

        self.encoder_layers = num_layers
        self.deg_dec_layers = deg_dec_layers
        self.fea_dec_layers = fea_dec_layers
        self.sample_size = sample_size
        self.sample_time = sample_time
        self.neigh_loss = neigh_loss
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3
        self.real_loss = real_loss
        self.neighbor_num_list = None
        self.verbose = verbose

    def process_graph(self, data):
        _, self.neighbor_num_list = GADNRBase.process_graph(data)
        self.neighbor_num_list = self.neighbor_num_list.to(self.device)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
                         
        return GADNRBase(in_dim=self.in_dim, hid_dim=self.hid_dim,
                         encoder_layers=self.encoder_layers,
                         deg_dec_layers=self.deg_dec_layers,
                         fea_dec_layers=self.fea_dec_layers,
                         sample_size=self.sample_size,
                         sample_time=self.sample_time,
                         batch_size=self.batch_size, 
                         neighbor_num_list=self.neighbor_num_list,
                         neigh_loss=self.neigh_loss,
                         lambda_loss1=self.lambda_loss1,
                         lambda_loss2=self.lambda_loss2,
                         lambda_loss3=self.lambda_loss3,
                         backbone=self.backbone,
                         device=self.device).to(self.device)

    def forward_model(self, data):
        
        if 'input_id' in data: # mini-batch training
            neighbor_dict, neighbor_num_list = GADNRBase.process_graph(data)
            # TODO rebuild the neighbor degree list
            self.neighbor_num_list = neighbor_num_list 
            h0, l1, degree_logits, feat_recon_list, neigh_recon_list = \
                                                self.model(data.x,
                                                           data.edge_index,
                                                           data.input_id,
                                                           neighbor_dict)
        else: # full batch training
            h0, l1, degree_logits, feat_recon_list, neigh_recon_list = \
                                                self.model(data.x,
                                                           data.edge_index)
        
        loss, loss_per_node, h_loss, degree_loss, feature_loss = \
                                self.model.loss_func(h0,
                                                     l1,
                                                     degree_logits,
                                                     feat_recon_list,
                                                     neigh_recon_list,
                                                     self.neighbor_num_list)

        return loss, loss_per_node.cpu().detach(), h_loss.cpu().detach(), \
            degree_loss.cpu().detach(), feature_loss.cpu().detach()
    
    def comp_decision_score(self,
                            loss_per_node,
                            h_loss,
                            degree_loss,
                            feature_loss,
                            h_loss_weight,
                            degree_loss_weight,
                            feature_loss_weight):
        """Compute the decision score based on orginal loss or adaptive loss.
        """
        if self.real_loss:
            # the orginal decision score from the loss
            comp_loss = loss_per_node
        else:
            # the adaptive decision score
            h_loss_norm = h_loss / (torch.max(h_loss) - 
                                    torch.min(h_loss))
            degree_loss_norm = degree_loss / \
                (torch.max(degree_loss) - torch.min(degree_loss))
            feature_loss_norm = feature_loss / \
                (torch.max(feature_loss) - torch.min(feature_loss))
            comp_loss = h_loss_weight * h_loss_norm \
                + degree_loss_weight *  degree_loss_norm \
                    + feature_loss_weight * feature_loss_norm
        return comp_loss

    def fit(self,
            data,
            label=None,
            h_loss_weight=1.0,
            degree_loss_weight=0.,
            feature_loss_weight=2.5,
            loss_step=20
            ):
        """
        Overwrite the base model fit function since GAD-NR uses 
        multiple personalized loss functions.
        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.
        h_loss_weight : float, optional
            The weight of the neighborhood reconstruction loss term used in 
            the adaptive decision score. Default: ``1.0``.
        degree_loss_weight : float, optional
            The weight of the node degree reconstruction loss term used in 
            the adaptive decision score. Default: ``0.``.
        feature_loss_weight : float, optional
            The weight of the node feature reconstruction loss term used in 
            the adaptive decision score. Default: ``2.5``.
        loss_step : int, optional
            The epoch interval to update the loss terms. Default: ``20``.
        """

        self.num_nodes, self.in_dim = data.x.shape
        self.process_graph(data)
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]
        else:
            loader = NeighborLoader(data,
                                    self.num_neigh,
                                    batch_size=self.batch_size)
        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        
        degree_params = list(map(id, self.model.degree_decoder.parameters()))
        base_params = filter(lambda p: id(p) not in degree_params,
                         self.model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params}, 
                                      {'params': self.model.degree_decoder.
                                       parameters(), 'lr': 1e-2}],
                                       lr=self.lr,
                                       weight_decay=self.weight_decay)
        
        min_loss = float('inf')
        self.arg_min_loss_per_node = None

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.epoch):
            start_time = time.time()
            epoch_loss = 0
            epoch_loss_per_node = torch.zeros(data.x.shape[0]) 
            if epoch%loss_step==0:
                self.model.lambda_loss2 = self.model.lambda_loss2 + 0.5
                self.model.lambda_loss3 = self.model.lambda_loss3 / 2
            
            # full batch training
            if self.batch_size == data.x.shape[0]:
                loss, loss_per_node, h_loss, degree_loss, feature_loss = \
                                                self.forward_model(data) 

                comp_loss = self.comp_decision_score(loss_per_node,
                                                     h_loss,
                                                     degree_loss,
                                                     feature_loss,
                                                     h_loss_weight,
                                                     degree_loss_weight,
                                                     feature_loss_weight)
                
                self.decision_score_ = comp_loss.squeeze(1)

                if self.save_emb:
                    self.emb = self.model.emb.cpu()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss = loss.item() * self.batch_size 
                epoch_loss_per_node = loss_per_node.squeeze(1)
            else: # mini batch training
                for sampled_data in loader:
                    batch_size = sampled_data.batch_size
                    node_idx = sampled_data.n_id

                    loss, loss_per_node, h_loss, degree_loss, feature_loss = \
                                            self.forward_model(sampled_data)
                    
                    comp_loss = self.comp_decision_score(loss_per_node,
                                                         h_loss,
                                                         degree_loss,
                                                         feature_loss,
                                                         h_loss_weight,
                                                         degree_loss_weight,
                                                         feature_loss_weight)
                                                
                    self.decision_score_[node_idx[:batch_size]] = \
                                                        comp_loss.squeeze(1)

                    if self.save_emb:
                        self.emb[node_idx[:batch_size]] = \
                            self.model.emb[:batch_size].cpu()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * batch_size
                    epoch_loss_per_node[node_idx[:batch_size]] = \
                                                    loss_per_node.squeeze(1)
            
            loss_value = epoch_loss / data.x.shape[0]

            if loss_value < min_loss:
                min_loss = loss_value
                self.arg_min_loss_per_node = epoch_loss_per_node
            
            logger(epoch=epoch,
                   loss=loss_value,
                   score=self.decision_score_,
                   target=label,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

        self._process_decision_score()
        return self
    
    def decision_function(self,
                          data,
                          label=None,
                          h_loss_weight=1.0,
                          degree_loss_weight=0.,
                          feature_loss_weight=2.5):
        """
        TODO update the full batch and mini batch inference 
        Overwrite the decision fuction from the base model due to the unique
        loss function and decision score from the GADNR paper.
        The three loss term weights must be the same as the fit function if
        ``real_loss`` is ``False``.
        """
        self.process_graph(data)
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_score = torch.zeros(data.x.shape[0])
        if self.save_emb:
            if type(self.hid_dim) is tuple:
                self.emb = (torch.zeros(data.x.shape[0], self.hid_dim[0]),
                            torch.zeros(data.x.shape[0], self.hid_dim[1]))
            else:
                self.emb = torch.zeros(data.x.shape[0], self.hid_dim)
        start_time = time.time()
        for sampled_data in loader:
            loss, loss_per_node, h_loss, degree_loss, feature_loss = \
                            self.forward_model(sampled_data)
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id
            if self.save_emb:
                if type(self.hid_dim) is tuple:
                    self.emb[0][node_idx[:batch_size]] = \
                        self.model.emb[0][:batch_size].cpu()
                    self.emb[1][node_idx[:batch_size]] = \
                        self.model.emb[1][:batch_size].cpu()
                else:
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()

            if self.real_loss:
                # the orginal decision score from the loss
                comp_loss = loss_per_node
            else:
                # the adaptive decision score
                h_loss_norm = h_loss / (torch.max(h_loss) - 
                                        torch.min(h_loss))
                degree_loss_norm = degree_loss / \
                    (torch.max(degree_loss) - torch.min(degree_loss))
                feature_loss_norm = feature_loss / \
                    (torch.max(feature_loss) - torch.min(feature_loss))
                comp_loss = h_loss_weight * h_loss_norm \
                    + degree_loss_weight *  degree_loss_norm \
                        + feature_loss_weight * feature_loss_norm
            
            outlier_score[node_idx[:batch_size]] = comp_loss.squeeze(1)

        logger(loss=loss.item() / data.x.shape[0],
               score=outlier_score,
               target=label,
               time=time.time() - start_time,
               verbose=self.verbose,
               train=False)
        return outlier_score
