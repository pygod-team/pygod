# -*- coding: utf-8 -*-
import os
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
from torch_geometric.nn import GCN
from torch_geometric.seed import seed_everything

from pygod.metric import eval_roc_auc
from pygod.detector import GADNR

seed_everything(717)


class TestGADNR(unittest.TestCase):
    def setUp(self):
        self.roc_floor = 0.60

        self.train_data = torch.load(os.path.join('pygod/test/train_graph.pt'))
        self.test_data = torch.load(os.path.join('pygod/test/test_graph.pt'))

    def test_full(self):
        detector = GADNR(epoch=5, num_layers=3)
        detector.fit(self.train_data)

        pred, score, conf = detector.predict(self.train_data,
                                             return_pred=True,
                                             return_score=True,
                                             return_conf=True)

        assert_equal(pred.shape[0], self.train_data.y.shape[0])
        assert (eval_roc_auc(self.train_data.y, score) >= self.roc_floor)
        assert_equal(conf.shape[0], self.train_data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)

        prob = detector.predict(self.train_data,
                                return_pred=False,
                                return_prob=True,
                                prob_method='linear')
        assert_equal(prob.shape[0], self.train_data.y.shape[0])
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

        prob = detector.predict(self.train_data,
                                return_pred=False,
                                return_prob=True,
                                prob_method='unify')
        assert_equal(prob.shape[0], self.train_data.y.shape[0])
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

        with assert_raises(ValueError):
            detector.predict(self.train_data,
                             return_prob=True,
                             prob_method='something')
        
        with assert_raises(ValueError):
            pred, score, conf, emb = detector.predict(self.test_data,
                                                      return_pred=True,
                                                      return_score=True,
                                                      return_conf=True,
                                                      return_emb=True)

    def test_sample(self):
        detector = GADNR(hid_dim=32,
                        num_layers=3,
                        deg_dec_layers=4,
                        fea_dec_layers=3,
                        backbone=GCN,
                        sample_size=2,
                        sample_time=3,
                        neigh_loss='KL',
                        lambda_loss1=0.01,
                        lambda_loss2=0.1,
                        lambda_loss3=0.8,
                        real_loss=True,
                        lr=0.01,
                        epoch=2,
                        dropout=0.1,
                        weight_decay=0.01,
                        act=torch.nn.functional.relu,
                        batch_size=16,
                        num_neigh=20,
                        contamination=0.2,
                        verbose=3,
                        save_emb=True)
        detector.fit(self.train_data)

        score = detector.predict(return_pred=False, return_score=True)
        # assert (eval_roc_auc(self.train_data.y, score) >= self.roc_floor)

        pred, score, conf, emb = detector.predict(self.test_data,
                                                  return_pred=True,
                                                  return_score=True,
                                                  return_conf=True,
                                                  return_emb=True)

        assert_equal(pred.shape[0], self.test_data.y.shape[0])
        # assert (eval_roc_auc(self.test_data.y, score) >= self.roc_floor)
        assert_equal(conf.shape[0], self.test_data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)
        assert_equal(emb.shape[0], self.test_data.y.shape[0])
        assert_equal(emb.shape[1], detector.hid_dim)

        prob = detector.predict(self.test_data,
                                return_pred=False,
                                return_prob=True,
                                prob_method='linear')
        assert_equal(prob.shape[0], self.test_data.y.shape[0])
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

        prob = detector.predict(self.test_data,
                                return_pred=False,
                                return_prob=True,
                                prob_method='unify')
        assert_equal(prob.shape[0], self.test_data.y.shape[0])
        assert (prob.min() >= 0)
        assert (prob.max() <= 1)

        with assert_raises(ValueError):
            detector.predict(self.test_data,
                             return_prob=True,
                             prob_method='something')

    def test_W2(self):
        detector = GADNR(hid_dim=32,
                        num_layers=3,
                        neigh_loss='W2',
                        epoch=2,
                        batch_size=16,
                        num_neigh=20,
                        verbose=3,
                        save_emb=True)
        detector.fit(self.train_data)

        score = detector.predict(return_pred=False, return_score=True)
        # assert (eval_roc_auc(self.train_data.y, score) >= self.roc_floor)

        pred, score, conf, emb = detector.predict(self.test_data,
                                                  return_pred=True,
                                                  return_score=True,
                                                  return_conf=True,
                                                  return_emb=True)

        assert_equal(pred.shape[0], self.test_data.y.shape[0])
        # assert (eval_roc_auc(self.test_data.y, score) >= self.roc_floor)
        assert_equal(conf.shape[0], self.test_data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)
        assert_equal(emb.shape[0], self.test_data.y.shape[0])
        assert_equal(emb.shape[1], detector.hid_dim)

    def test_comp_loss(self):
        detector = GADNR(epoch=5, num_layers=3, real_loss=False)
        detector.fit(self.train_data)

        pred, score, conf = detector.predict(self.train_data,
                                             return_pred=True,
                                             return_score=True,
                                             return_conf=True)

        assert_equal(pred.shape[0], self.train_data.y.shape[0])
        assert_equal(conf.shape[0], self.train_data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)
    
    def test_params(self):
        with assert_raises(ValueError):
            detector = GADNR(epoch=5, num_layers=3, neigh_loss='something')
            detector.fit(self.test_data)
