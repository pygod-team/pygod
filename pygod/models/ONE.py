import pandas as pd
import numpy as np
import time
from sklearn.decomposition import NMF
import sys
import pickle
import networkx as nx
import os
import gc
import os.path as osp

from sklearn.utils.validation import check_is_fitted

import argparse

from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import to_scipy_sparse_matrix

from base import BaseDetector

gc.enable()

class ONE(BaseDetector):
    """Let us decide the documentation later
    ONE (Outlier Aware Network Embedding for Attributed Networks)
    Reference: <https://ojs.aaai.org//index.php/AAAI/article/view/3763>
    """
    
    def __init__(self, contamination=0.1):
        super(ONE, self).__init__(contamination=contamination)

    def fit(self, Graph, args):
        A, C, true_labels = self.process_graph(Graph, args)

        NoComm = len(set(true_labels))

        assert (A.shape[0] == C.shape[0] & A.shape[0] == len(true_labels))

        K = args.K

        print("Number of Dimensions : ", K)
        self.W = np.eye(K)

        print('Dimension of C: {}, {}'.format(C.shape[0], C.shape[1]))
        gc.collect()
        opti_values = []
        runtime = []
        self.mu = 1
        gc.collect()
        start_time = time.time()

        model = NMF(n_components=K, init='random', random_state=0)
        self.G = model.fit_transform(A)
        self.H = model.components_

        model = NMF(n_components=K, init='random', random_state=0)
        self.U = model.fit_transform(C)
        self.V = model.components_

        outl1 = np.ones((A.shape[0]))
        outl2 = np.ones((A.shape[0]))
        outl3 = np.ones((A.shape[0]))
        Graph = nx.from_numpy_matrix(A)
        bet = nx.betweenness_centrality(Graph)
        for i in range(len(outl1)):
            outl1[i] = float(1) / A.shape[0] + bet[i]
            outl2[i] = float(1) / A.shape[0]
            outl3[i] = float(1) / A.shape[0] + bet[i]

        outl1 = outl1 / sum(outl1)
        outl2 = outl2 / sum(outl2)
        outl3 = outl3 / sum(outl3)

        count_outer = args.iter  # Number of outer Iterations for optimization

        temp1 = A - np.matmul(self.G, self.H)
        temp1 = np.multiply(temp1, temp1)
        temp1 = np.multiply(np.log(np.reciprocal(outl1)), np.sum(temp1, axis=1))
        temp1 = np.sum(temp1)

        temp2 = C - np.matmul(self.U, self.V)
        temp2 = np.multiply(temp2, temp2)
        temp2 = np.multiply(np.log(np.reciprocal(outl2)), np.sum(temp2, axis=1))
        temp2 = np.sum(temp2)

        temp3 = self.G.T - np.matmul(self.W, self.U.T)
        temp3 = np.multiply(temp3, temp3)
        temp3 = np.multiply(np.log(np.reciprocal(outl3)), np.sum(temp3, axis=0).T)
        temp3 = np.sum(temp3)

        self.alpha = 1
        self.beta = temp1 / temp2
        self.gamma = min(2 * self.beta, temp3)

        for opti_iter in range(count_outer):

            print('Loop {} started: \n'.format(opti_iter))

            print("The function values which we are interested are : ")

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)

            # The Updation rule for G[i,k]
            for i in range(self.G.shape[0]):
                for k in range(self.G.shape[1]):
                    Gik_numer = self.alpha * np.log(np.reciprocal(outl1[i])) * np.dot(self.H[k, :], \
                                                                                 (A[i, :] - (np.matmul(self.G[i],
                                                                                                       self.H) - np.multiply(
                                                                                     self.G[i, k], self.H[k, :])))) + \
                                self.gamma * np.log(np.reciprocal(outl3[i])) * np.dot(self.U[i], self.W[k, :])
                    Gik_denom = self.alpha * np.log(np.reciprocal(outl1[i])) * np.dot(self.H[k, :], self.H[k, :]) + \
                                self.gamma * np.log(np.reciprocal(outl3[i]))

                    self.G[i, k] = Gik_numer / Gik_denom

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)
            print('Done for G')

            # The updation rule for H[k,j]
            for k in range(self.H.shape[0]):
                for j in range(self.H.shape[1]):
                    Hkj_numer = self.alpha * np.dot(np.multiply(np.log(np.reciprocal(outl1)), self.G[:, k]), \
                                               (A[:, j] - (np.matmul(self.G, self.H[:, j]) - np.multiply(self.G[:, k],
                                                                                               self.H[k, j]))))
                    Hkj_denom = self.alpha * (np.dot(np.log(np.reciprocal(outl1)), np.multiply(self.G[:, k], self.G[:, k])))

                    self.H[k, j] = Hkj_numer / Hkj_denom

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)
            print('Done for H')

            # The up[dation rule for U[i,k]
            for i in range(self.U.shape[0]):
                for k in range(self.U.shape[1]):
                    Uik_numer_1 = self.beta * np.log(np.reciprocal(outl2[i])) * (np.dot(self.V[k, :], \
                                                                                   (C[i] - (np.matmul(self.U[i, :],
                                                                                                      self.V) - np.multiply(
                                                                                       self.U[i, k], self.V[k, :])))))

                    Uik_numer_2 = self.gamma * np.log(np.reciprocal(outl3[i])) * np.dot( \
                        (self.G[i, :] - (np.matmul(self.U[i, :], self.W) - np.multiply(self.U[i, k], self.W[:, k]))), self.W[:, k])

                    Uik_denom = self.beta * np.log(np.reciprocal(outl2[i])) * np.dot(self.V[k, :], self.V[k, :] \
                                                                                ) + self.gamma * np.log(
                        np.reciprocal(outl3[i])) * np.dot(self.W[:, k], self.W[:, k])

                    self.U[i, k] = (Uik_numer_1 + Uik_numer_2) / Uik_denom

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)
            print('Done for U')

            # The updation rule for V[k,d]
            for k in range(self.V.shape[0]):
                for d in range(self.V.shape[1]):
                    Vkd_numer = self.beta * np.dot(np.multiply(np.log(np.reciprocal(outl2)), self.U[:, k]), (C[:, d] \
                                                                                                   - (np.matmul(
                                self.U,
                                self.V[:,
                                d]) - np.multiply(
                                self.U[:, k], self.V[k, d]))))
                    Vkd_denom = self.beta * (np.dot(np.log(np.reciprocal(outl2)), np.multiply(self.U[:, k], self.U[:, k])))

                    self.V[k][d] = Vkd_numer / Vkd_denom

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)
            print('Done for V')

            # The Update rule for W[p,q]

            logoi = np.log(np.reciprocal(outl3))
            sqrt_logoi = np.sqrt(logoi)
            sqrt_logoi = np.tile(sqrt_logoi, (K, 1))
            assert (sqrt_logoi.shape == self.G.T.shape)

            term1 = np.multiply(sqrt_logoi, self.G.T)
            term2 = np.multiply(sqrt_logoi, self.U.T)

            svd_matrix = np.matmul(term1, term2.T)

            svd_u, svd_sigma, svd_vt = np.linalg.svd(svd_matrix)

            self.W = np.matmul(svd_u, svd_vt)

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)
            print('Done for W')

            # The updation rule for outl

            outl1, outl2, outl3 = self.cal_outlierScore(A, C)

            self.calc_lossValues(A, C, self.G, self.H, self.U, self.V, self.W, outl1, outl2, outl3, self.alpha, self.beta, self.gamma)
            print('Done for outlier score')

            print('Loop {} ended: \n'.format(opti_iter))

            # with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'G', 'wb') as file:
            #     pickle.dump(G, file)
            # with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'U', 'wb') as file:
            #     pickle.dump(U, file)
            # with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'W', 'wb') as file:
            #     pickle.dump(W, file)
            # with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'Oi1',
            #           'wb') as file:
            #     pickle.dump(outl1, file)
            # with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'Oi2',
            #           'wb') as file:
            #     pickle.dump(outl2, file)
            # with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'Oi3',
            #           'wb') as file:
            #     pickle.dump(outl3, file)

        # Use outl2 as the outlier score.
        # In the paper: "We have observed experimentally thatO2is more important to determine out-liers."
        self.decision_scores_ = outl2
        self._process_decision_scores()

        return self

    def decision_function(self, Graph, args):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['W', 'G', 'H', 'U', 'V'])

        A, C, true_labels = self.process_graph(Graph, args)

        _, outl2, _ = self.cal_outlierScore(A, C)

        return outl2

    def process_graph(self, Graph, args):
        """

        :param Graph:
        :param args:
        :return:
        """

        A = to_scipy_sparse_matrix(Graph['edge_index']).toarray().astype('float64')

        C = Graph['x'].numpy().astype('float64')

        true_labels = Graph['y'].tolist()

        return A, C, true_labels

    def calc_lossValues(self, A, C, G, H, U, V, W, outl1, outl2, outl3, alpha, beta, gamma):
        temp1 = A - np.matmul(G, H)
        temp1 = np.multiply(temp1, temp1)
        temp1 = np.multiply(np.log(np.reciprocal(outl1)), np.sum(temp1, axis=1))
        temp1 = np.sum(temp1)

        temp2 = C - np.matmul(U, V)
        temp2 = np.multiply(temp2, temp2)
        temp2 = np.multiply(np.log(np.reciprocal(outl2)), np.sum(temp2, axis=1))
        temp2 = np.sum(temp2)

        temp3 = G.T - np.matmul(W, U.T)
        temp3 = np.multiply(temp3, temp3)
        temp3 = np.multiply(np.log(np.reciprocal(outl3)), np.sum(temp3, axis=0).T)
        temp3 = np.sum(temp3)

        print('\t Component values: {},{} and {}'.format(temp1, temp2, temp3))

        func_value = alpha * temp1 + beta * temp2 + gamma * temp3

        print('\t Total Function value {}'.format(func_value))

    def cal_outlierScore(self, A, C):
        GH = np.matmul(self.G, self.H)
        UV = np.matmul(self.U, self.V)
        WUTrans = np.matmul(self.W, self.U.T)

        outl1_numer = self.alpha * (np.multiply((A - GH), (A - GH))).sum(axis=1)

        outl1_denom = self.alpha * pow(np.linalg.norm((A - GH), 'fro'), 2)

        outl1_numer = outl1_numer * self.mu
        outl1 = outl1_numer / outl1_denom

        outl2_numer = self.beta * (np.multiply((C - UV), (C - UV))).sum(axis=1)

        outl2_denom = self.beta * pow(np.linalg.norm((C - UV), 'fro'), 2)

        outl2_numer = outl2_numer * self.mu
        outl2 = outl2_numer / outl2_denom

        outl3_numer = self.gamma * (np.multiply((self.G.T - WUTrans), (self.G.T - WUTrans))).sum(axis=0).T

        outl3_denom = self.gamma * pow(np.linalg.norm((self.G.T - WUTrans), 'fro'), 2)

        outl3_numer = outl3_numer * self.mu
        outl3 = outl3_numer / outl3_denom

        return outl1, outl2, outl3


if __name__ == '__main__':
    # todo: need a default args template
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CiteSeer',
                        help='dataset name: CiteSeer')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='dimension of hidden embedding (default: 64)')
    parser.add_argument('--K', type=int, default=36,
                        help='map every vertex to a K dimensional vector, where K < min(N,D). (default: 64)')
    parser.add_argument('--iter', type=int, default=5,
                        help='Number of training iterations for optimization. (default: 5)')
    parser.add_argument('--epoch', type=int, default=3, help='Training epoch')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--device', default='cpu', type=str, help='cuda/cpu')

    args = parser.parse_args()

    # data loading
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    args.dataset)

    # this gives us a PyG data object
    Graph = AttributedGraphDataset(path, 'CiteSeer')[0]

    # model initialization
    clf = ONE()

    print('training...')
    clf.fit(Graph, args)
    print()

    print('predicting for probability')
    prob = clf.predict_proba(Graph, args)
    print('Probability', prob)
    print()

    print('predicting for raw scores')
    outlier_scores = clf.decision_function(Graph, args)
    print('Raw scores', outlier_scores)
    print()

    print('predicting for labels')
    labels = clf.predict(Graph, args)
    print('Labels', labels)
    print()

    print('predicting for labels with confidence')
    labels, confidence = clf.predict(Graph, args, return_confidence=True)
    print('Labels', labels)
    print('Confidence', confidence)
    print()
