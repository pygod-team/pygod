# -*- coding: utf-8 -*-
"""Outlier Aware Network Embedding for Attributed Networks (ONE)
"""
# Author: Xiyang Hu <xiyanghu@cmu.edu>
# License: BSD 2 clause

import time
import gc
import numpy as np
import networkx as nx
from sklearn.decomposition import NMF

from sklearn.utils.validation import check_is_fitted
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx

from . import BaseDetector
from ..utils.metric import eval_roc_auc

gc.enable()


# todo: to optimize later
# @njit
def calculate_G(G_mat, alpha, outl1, H, A, gamma, outl3, U, W):
    # The update rule for G_mat[i,k]
    for i in range(G_mat.shape[0]):
        for k in range(G_mat.shape[1]):
            Gik_numer = alpha * np.log(
                np.reciprocal(outl1[i])) * np.dot(H[k, :], (
                    A[i, :] - (
                    np.matmul(G_mat[i], H) - np.multiply(
                G_mat[i, k], H[k, :])))) \
                        + gamma * np.log(
                np.reciprocal(outl3[i])) * np.dot(U[i],
                                                  W[k, :])

            Gik_denom = alpha * np.log(
                np.reciprocal(outl1[i])) * np.dot(H[k, :],
                                                  H[k, :]) + \
                        gamma * np.log(np.reciprocal(outl3[i]))

            G_mat[i, k] = Gik_numer / Gik_denom
            return G_mat


# todo: due to the original paper has very complex loss, this algorithm is not
# in PyTorch yet. Need NetworkX for it.
class ONE(BaseDetector):
    """
    ONE (Outlier Aware Network Embedding for Attributed Networks)
    Reference: <https://arxiv.org/pdf/1811.07609.pdf>


    See :cite:`bandyopadhyay2019outlier` for details.

    Parameters
    ----------
    K :  int, optional
        Every vertex is a K dimensional vector, K < min(N, D). Default: ``36``.
    iter : int, optional
        Number of outer Iterations for optimization.  Default: ``5``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import ONE
    >>> model = ONE()
    >>> model.fit(data) # PyG graph data object
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 K=36,
                 iter=5,
                 contamination=0.1,
                 verbose=False):
        super(ONE, self).__init__(contamination=contamination)

        self.K = K
        self.iter = iter

        # other param
        self.verbose = verbose

    def fit(self, G, y_true=None):
        """
        Description
        -----------
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        y_true : numpy.array, optional (default=None)
            The optional outlier ground truth labels used to monitor the
            training progress. They are not used to optimize the
            unsupervised model.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        A, C = self.process_graph(G)

        assert A.shape[0] == C.shape[0]

        K = self.K

        if self.verbose:
            print("Number of Dimensions : ", K)
        self.W = np.eye(K)

        if self.verbose:
            print('Dimension of C: {}, {}'.format(C.shape[0], C.shape[1]))
        gc.collect()
        opti_values = []
        runtime = []
        self.mu = 1
        gc.collect()
        start_time = time.time()

        model = NMF(n_components=K, init='random', random_state=0)
        # ensure A is non-negative
        if A.min() < 0:
            A = A + A.min() * -1
        self.G_mat = model.fit_transform(A)
        self.H = model.components_

        model = NMF(n_components=K, init='random', random_state=0)
        # ensure C is non-negative
        if C.min() < 0:
            C = C + C.min() * -1
        self.U = model.fit_transform(C)
        self.V = model.components_

        outl1 = outl2 = outl3 = np.ones((A.shape[0]))

        G = to_networkx(G)
        bet = nx.betweenness_centrality(G)
        for i in range(len(outl1)):
            outl1[i] = float(1) / A.shape[0] + bet[i]
            outl2[i] = float(1) / A.shape[0]
            outl3[i] = float(1) / A.shape[0] + bet[i]

        outl1 = outl1 / sum(outl1)
        outl2 = outl2 / sum(outl2)
        outl3 = outl3 / sum(outl3)

        count_outer = self.iter

        temp1 = A - np.matmul(self.G_mat, self.H)
        temp1 = np.multiply(temp1, temp1)
        temp1 = np.multiply(np.log(np.reciprocal(outl1)),
                            np.sum(temp1, axis=1))
        temp1 = np.sum(temp1)

        temp2 = C - np.matmul(self.U, self.V)
        temp2 = np.multiply(temp2, temp2)
        temp2 = np.multiply(np.log(np.reciprocal(outl2)),
                            np.sum(temp2, axis=1))
        temp2 = np.sum(temp2)

        temp3 = self.G_mat.T - np.matmul(self.W, self.U.T)
        temp3 = np.multiply(temp3, temp3)
        temp3 = np.multiply(np.log(np.reciprocal(outl3)),
                            np.sum(temp3, axis=0).T)
        temp3 = np.sum(temp3)

        self.alpha = 1
        self.beta = temp1 / temp2
        self.gamma = min(2 * self.beta, temp3)

        for opti_iter in range(count_outer):
            if self.verbose:
                print('Loop {} started: \n'.format(opti_iter))
                print("The function values which we are interested are : ")

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)

            # The update rule for G_mat[i,k]
            for i in range(self.G_mat.shape[0]):
                for k in range(self.G_mat.shape[1]):
                    Gik_numer = self.alpha * np.log(
                        np.reciprocal(outl1[i])) * np.dot(self.H[k, :], (
                            A[i, :] - (
                            np.matmul(self.G_mat[i], self.H) - np.multiply(
                        self.G_mat[i, k], self.H[k, :])))) \
                                + self.gamma * np.log(
                        np.reciprocal(outl3[i])) * np.dot(self.U[i],
                                                          self.W[k, :])

                    Gik_denom = self.alpha * np.log(
                        np.reciprocal(outl1[i])) * np.dot(self.H[k, :],
                                                          self.H[k, :]) + \
                                self.gamma * np.log(np.reciprocal(outl3[i]))

                    self.G_mat[i, k] = Gik_numer / Gik_denom

            # self.G_mat = calculate_G(self.G_mat, self.alpha, outl1,
            #              self.H, A, self.gamma, outl3, self.U, self.W)

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)
            if self.verbose:
                print('Done for G_mat')

            # The update rule for H[k,j]
            for k in range(self.H.shape[0]):
                for j in range(self.H.shape[1]):
                    Hkj_numer = self.alpha * np.dot(
                        np.multiply(np.log(np.reciprocal(outl1)),
                                    self.G_mat[:, k]), \
                        (A[:, j] - (np.matmul(self.G_mat,
                                              self.H[:, j]) - np.multiply(
                            self.G_mat[:, k],
                            self.H[k, j]))))
                    Hkj_denom = self.alpha * (
                        np.dot(np.log(np.reciprocal(outl1)),
                               np.multiply(self.G_mat[:, k],
                                           self.G_mat[:, k])))

                    self.H[k, j] = Hkj_numer / Hkj_denom

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)
            if self.verbose:
                print('Done for H')

            # The up[update rule for U[i,k]
            for i in range(self.U.shape[0]):
                for k in range(self.U.shape[1]):
                    Uik_numer_1 = self.beta * np.log(
                        np.reciprocal(outl2[i])) * \
                                  (np.dot(self.V[k, :],
                                          (C[i] - (np.matmul(
                                              self.U[i, :],
                                              self.V) - np.multiply(
                                              self.U[i, k],
                                              self.V[k, :])))))

                    Uik_numer_2 = self.gamma * np.log(
                        np.reciprocal(outl3[i])) * np.dot(
                        (self.G_mat[i, :] - (np.matmul(self.U[i, :],
                                                       self.W) - np.multiply(
                            self.U[i, k], self.W[:, k]))), self.W[:, k])

                    Uik_denom = self.beta * np.log(
                        np.reciprocal(outl2[i])) * np.dot(self.V[k, :],
                                                          self.V[k, :]
                                                          ) + self.gamma * \
                                np.log(np.reciprocal(outl3[i])) * np.dot(
                        self.W[:, k],
                        self.W[:, k])

                    self.U[i, k] = (Uik_numer_1 + Uik_numer_2) / Uik_denom

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)
            if self.verbose:
                print('Done for U')

            # The update rule for V[k,d]
            for k in range(self.V.shape[0]):
                for d in range(self.V.shape[1]):
                    Vkd_numer = self.beta * np.dot(
                        np.multiply(np.log(np.reciprocal(outl2)),
                                    self.U[:, k]), (C[:, d]
                                                    - (np.matmul(
                                        self.U,
                                        self.V[:, d]) - np.multiply(
                                        self.U[:, k], self.V[k, d]))))
                    Vkd_denom = self.beta * (
                        np.dot(np.log(np.reciprocal(outl2)),
                               np.multiply(self.U[:, k], self.U[:, k])))

                    self.V[k][d] = Vkd_numer / Vkd_denom

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)
            if self.verbose:
                print('Done for V')

            # The Update rule for W[p,q]

            logoi = np.log(np.reciprocal(outl3))
            sqrt_logoi = np.sqrt(logoi)
            sqrt_logoi = np.tile(sqrt_logoi, (K, 1))
            assert (sqrt_logoi.shape == self.G_mat.T.shape)

            term1 = np.multiply(sqrt_logoi, self.G_mat.T)
            term2 = np.multiply(sqrt_logoi, self.U.T)

            svd_matrix = np.matmul(term1, term2.T)

            svd_u, svd_sigma, svd_vt = np.linalg.svd(svd_matrix)

            self.W = np.matmul(svd_u, svd_vt)

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)
            if self.verbose:
                print('Done for W')

            # The update rule for outl

            outl1, outl2, outl3 = self.cal_outlierScore(A, C)

            self.calc_lossValues(A, C, self.G_mat, self.H, self.U, self.V,
                                 self.W,
                                 outl1, outl2, outl3, self.alpha, self.beta,
                                 self.gamma)
            if self.verbose:
                print('Done for outlier score')
                print('Loop {} ended: \n'.format(opti_iter))

        if self.verbose:
            if y_true is not None:
                auc = eval_roc_auc(y_true, outl2)
                print(" | AUC {:.4f}".format(auc), end='')
            print()

        # Use outl2 as the outlier score.
        # In the paper:
        # "O2 is more important to determine outliers."
        self.decision_scores_ = outl2
        self._process_decision_scores()

        return self

    def decision_function(self, G):
        """
        Description
        -----------
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outl2 : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['W', 'G_mat', 'H', 'U', 'V'])

        A, C = self.process_graph(G)
        C = C - C.min() + 1  # NMF requires the matrix to be non-negative.

        _, outl2, _ = self.cal_outlierScore(A, C)

        return outl2

    def process_graph(self, G):
        """
        Description
        -----------
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input graph.

        Returns
        -------
        A : numpy.array
            The adjacency matrix.
        C : numpy.array
            The node attribute matrix.
        """

        # todo: need some assert or try/catch to make sure certain attributes
        # are presented.

        A = to_scipy_sparse_matrix(G['edge_index']).toarray().astype(
            'float64')

        C = G['x'].numpy().astype('float64')

        # return data objects needed for the network
        return A, C

    def calc_lossValues(self,
                        A,
                        C,
                        G_mat,
                        H,
                        U,
                        V,
                        W,
                        outl1,
                        outl2,
                        outl3,
                        alpha,
                        beta,
                        gamma):
        """
        Description
        -----------
        Calculate the loss. This function is called inside the fit() function.

        Parameters
        ----------
        Multiple variables inside the fit() function.

        Returns
        -------
        None
        """
        temp1 = A - np.matmul(G_mat, H)
        temp1 = np.multiply(temp1, temp1)
        temp1 = np.multiply(np.log(np.reciprocal(outl1)),
                            np.sum(temp1, axis=1))
        temp1 = np.sum(temp1)

        temp2 = C - np.matmul(U, V)
        temp2 = np.multiply(temp2, temp2)
        temp2 = np.multiply(np.log(np.reciprocal(outl2)),
                            np.sum(temp2, axis=1))
        temp2 = np.sum(temp2)

        temp3 = G_mat.T - np.matmul(W, U.T)
        temp3 = np.multiply(temp3, temp3)
        temp3 = np.multiply(np.log(np.reciprocal(outl3)),
                            np.sum(temp3, axis=0).T)
        temp3 = np.sum(temp3)
        if self.verbose:
            print('\t Component values: {},{} and {}'.format(temp1, temp2,
                                                             temp3))

        func_value = alpha * temp1 + beta * temp2 + gamma * temp3
        if self.verbose:
            print('\t Total Function value {}'.format(func_value))

    def cal_outlierScore(self,
                         A,
                         C):
        """
        Description
        -----------
        Calculate the outlier scores.

        Parameters
        ----------
        A : numpy.array
            The adjacency matrix.
        C : numpy.array
            The node attribute matrix.

        Returns
        -------
        outlier_scores : Tuple(numpy.array, numpy.array, numpy.array)
            Three sets of outlier scores from three different layers.
        """
        GH = np.matmul(self.G_mat, self.H)
        UV = np.matmul(self.U, self.V)
        WUTrans = np.matmul(self.W, self.U.T)

        outl1_numer = self.alpha * (np.multiply((A - GH), (A - GH))).sum(
            axis=1)

        outl1_denom = self.alpha * pow(np.linalg.norm((A - GH), 'fro'), 2)

        outl1_numer *= self.mu
        outl1 = outl1_numer / outl1_denom

        outl2_numer = self.beta * (np.multiply((C - UV), (C - UV))).sum(axis=1)

        outl2_denom = self.beta * pow(np.linalg.norm((C - UV), 'fro'), 2)

        outl2_numer *= self.mu
        outl2 = outl2_numer / outl2_denom

        outl3_numer = self.gamma * (
            np.multiply((self.G_mat.T - WUTrans),
                        (self.G_mat.T - WUTrans))).sum(
            axis=0).T

        outl3_denom = self.gamma * pow(
            np.linalg.norm((self.G_mat.T - WUTrans), 'fro'), 2)

        outl3_numer *= self.mu
        outl3 = outl3_numer / outl3_denom

        return outl1, outl2, outl3
