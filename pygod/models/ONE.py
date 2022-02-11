
# TODO: (1) data reading - change into load data from pyg;
#  (2) restructure the model into forward

import pandas as pd
import numpy as np
import time
from sklearn.decomposition import NMF
import sys
import pickle
import networkx as nx
import os
import gc

import torch

gc.enable()

mainpath = os.path.dirname(os.path.realpath(sys.argv[0]))
mainpath = mainpath + '/Data'


def calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3):
    temp1 = A - torch.matmul(G, H)
    temp1 = torch.mul(temp1, temp1)
    temp1 = torch.mul(torch.log(torch.reciprocal(outl1)), torch.sum(temp1, 1))
    temp1 = torch.sum(temp1)

    temp2 = C - torch.matmul(U, V)
    temp2 = torch.mul(temp2, temp2)
    temp2 = torch.mul(torch.log(torch.reciprocal(outl2)), torch.sum(temp2, 1))
    temp2 = torch.sum(temp2)

    temp3 = G.transpose(0, 1) - torch.matmul(W, U.transpose(0, 1))
    temp3 = torch.mul(temp3, temp3)
    temp3 = torch.mul(torch.log(torch.reciprocal(outl3)), torch.sum(temp3, 0))
    temp3 = torch.sum(temp3)

    print('\t Component values: {},{} and {}'.format(temp1, temp2, temp3))

    func_value = alpha * temp1 + beta * temp2 + gamma * temp3

    print('\t Total Function value {}'.format(func_value))


# NOTE : A node is an outlier ie(the outlier we seeded) when permutation[node_index in struct] > 3312 for citeseer
# NOTE : A node is an outlier ie(the outlier we seeded) when permutation[node_index in struct] > 877 for citeseer


DSNames = ['citeseer']

for datasetname in DSNames:

    result_dir_original = mainpath + '/' + datasetname + '_results/' + 'citeseer/'

    if not os.path.exists(result_dir_original):
        os.makedirs(result_dir_original)
        os.makedirs(result_dir_original + "k=2")
        os.makedirs(result_dir_original + "k=3")
        os.makedirs(result_dir_original + "k=4")
        os.makedirs(result_dir_original + "k=5")
        os.makedirs(result_dir_original + "k=6")

    print("Check your result dir" + result_dir_original)

    print('Dataset: {}'.format(datasetname))

    filepath = mainpath + '/citeseer/'
    structfile = 'struct.csv'
    contfile = 'content.csv'
    fileLabels = 'label.csv'

    A = pd.read_csv(filepath + structfile, header=None)
    A = torch.tensor(A.values)

    C = pd.read_csv(filepath + contfile, header=None)
    C = torch.tensor(C.values)

    true_labels = pd.read_csv(filepath + fileLabels, header=None)
    true_labels = true_labels[0].tolist()  # TODO

    NoComm = len(set(true_labels))

    assert (A.shape[0] == C.shape[0] & A.shape[0] == len(true_labels))

    for k_iter in [2, 3, 4, 5, 6]:
        K = k_iter * NoComm

        result_dir = result_dir_original + 'k=' + str(k_iter) + '/'
        sys.stdout.flush()
        sys.stdout = open(result_dir + 'results.txt', "a")

        print("Number of Dimensions : ", K)
        W = torch.eye(K, dtype=torch.double)

        print('Dimension of C: {}, {}'.format(C.shape[0], C.shape[1]))
        gc.collect()
        opti_values = []
        runtime = []
        mu = 1
        gc.collect()
        start_time = time.time()

        model = NMF(n_components=K, init='random', random_state=0)
        G = torch.tensor(model.fit_transform(A))
        H = torch.tensor(model.components_)

        model = NMF(n_components=K, init='random', random_state=0)
        U = torch.tensor(model.fit_transform(C))
        V = torch.tensor(model.components_)

        outl1 = np.ones((A.shape[0]))
        outl2 = np.ones((A.shape[0]))
        outl3 = np.ones((A.shape[0]))
        Graph = nx.from_numpy_matrix(A.numpy())
        bet = nx.betweenness_centrality(Graph)
        for i in range(len(outl1)):
            outl1[i] = float(1) / A.shape[0] + bet[i]
            outl2[i] = float(1) / A.shape[0]
            outl3[i] = float(1) / A.shape[0] + bet[i]

        outl1 = outl1 / sum(outl1)
        outl2 = outl2 / sum(outl2)
        outl3 = outl3 / sum(outl3)

        outl1 = torch.tensor(outl1)
        outl2 = torch.tensor(outl2)
        outl3 = torch.tensor(outl3)

        count_outer = 5  # Number of outer Iterations for optimization

        temp1 = A - torch.matmul(G, H)
        temp1 = torch.mul(temp1, temp1)
        temp1 = torch.mul(torch.log(torch.reciprocal(outl1)), torch.sum(temp1, 1))
        temp1 = torch.sum(temp1)

        temp2 = C - torch.matmul(U, V)
        temp2 = torch.mul(temp2, temp2)
        temp2 = torch.mul(torch.log(torch.reciprocal(outl2)), torch.sum(temp2, 1))
        temp2 = torch.sum(temp2)

        print(">>> G: ", G)
        print(">>> W: ", W)
        print(">>> U: ", U)
        temp3 = G.transpose(0, 1) - torch.matmul(W, U.transpose(0, 1))
        temp3 = torch.mul(temp3, temp3)
        print(">>> temp3.shape: ", temp3.shape)
        temp3 = torch.mul(torch.log(torch.reciprocal(outl3)), torch.sum(temp3, 0))
        temp3 = torch.sum(temp3)

        alpha = 1
        beta = temp1 / temp2
        gamma = min(2 * beta, temp3)

        for passNo in range(1):  # mu = 1 is good enough for all datasets

            print('Pass {} Started'.format(passNo))

            for opti_iter in range(count_outer):

                print('Loop {} started: \n'.format(opti_iter))

                print("The function values which we are interested are : ")

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)

                # The Updation rule for G[i,k]
                for i in range(G.shape[0]):
                    for k in range(G.shape[1]):
                        Gik_numer = alpha * torch.log(torch.reciprocal(outl1[i])) * torch.dot(H[k, :], \
                                                                                     (A[i, :] - (torch.matmul(G[i],
                                                                                                           H) - torch.mul(
                                                                                         G[i, k], H[k, :])))) + \
                                    gamma * torch.log(torch.reciprocal(outl3[i])) * torch.dot(U[i], W[k, :])
                        Gik_denom = alpha * torch.log(torch.reciprocal(outl1[i])) * torch.dot(H[k, :], H[k, :]) + \
                                    gamma * torch.log(torch.reciprocal(outl3[i]))

                        G[i, k] = Gik_numer / Gik_denom

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)
                print('Done for G')

                # The updation rule for H[k,j]
                for k in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        Hkj_numer = alpha * torch.dot(torch.multiply(torch.log(torch.reciprocal(outl1)), G[:, k]), \
                                                   (A[:, j] - (torch.matmul(G, H[:, j]) - torch.multiply(G[:, k], H[k, j]))))
                        Hkj_denom = alpha * (torch.dot(torch.log(torch.reciprocal(outl1)), torch.multiply(G[:, k], G[:, k])))

                        H[k, j] = Hkj_numer / Hkj_denom

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)
                print('Done for H')

                # The up[dation rule for U[i,k]
                for i in range(U.shape[0]):
                    for k in range(U.shape[1]):
                        Uik_numer_1 = beta * torch.log(torch.reciprocal(outl2[i])) * (torch.dot(V[k, :], \
                                                                                       (C[i] - (torch.matmul(U[i, :],
                                                                                                          V) - torch.multiply(
                                                                                           U[i, k], V[k, :])))))

                        Uik_numer_2 = gamma * torch.log(torch.reciprocal(outl3[i])) * torch.dot( \
                            (G[i, :] - (torch.matmul(U[i, :], W) - torch.multiply(U[i, k], W[:, k]))), W[:, k])

                        Uik_denom = beta * torch.log(torch.reciprocal(outl2[i])) * torch.dot(V[k, :], V[k, :] \
                                                                                    ) + gamma * torch.log(
                            torch.reciprocal(outl3[i])) * torch.dot(W[:, k], W[:, k])

                        U[i, k] = (Uik_numer_1 + Uik_numer_2) / Uik_denom

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)
                print('Done for U')

                # The updation rule for V[k,d]
                for k in range(V.shape[0]):
                    for d in range(V.shape[1]):
                        Vkd_numer = beta * torch.dot(torch.multiply(torch.log(torch.reciprocal(outl2)), U[:, k]), (C[:, d] \
                                                                                                       - (torch.matmul(U,
                                                                                                                    V[:,
                                                                                                                    d]) - torch.multiply(
                                    U[:, k], V[k, d]))))
                        Vkd_denom = beta * (torch.dot(torch.log(torch.reciprocal(outl2)), torch.multiply(U[:, k], U[:, k])))

                        V[k][d] = Vkd_numer / Vkd_denom

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)
                print('Done for V')

                # The Update rule for W[p,q]

                logoi = torch.log(torch.reciprocal(outl3))
                sqrt_logoi = torch.sqrt(logoi)
                sqrt_logoi = torch.tile(sqrt_logoi, (K, 1))
                assert (sqrt_logoi.shape == G.transpose(0, 1).shape)

                term1 = torch.mul(sqrt_logoi, G.transpose(0, 1))
                term2 = torch.mul(sqrt_logoi, U.transpose(0, 1))

                svd_matrix = torch.matmul(term1, term2.transpose(0, 1))

                svd_u, svd_sigma, svd_vt = torch.svd(svd_matrix)

                W = torch.matmul(svd_u, svd_vt)

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)
                print('Done for W')

                # The updation rule for outl

                GH = torch.matmul(G, H)
                UV = torch.matmul(U, V)
                WUTrans = torch.matmul(W, U.transpose(0, 1))

                outl1_numer = alpha * (torch.multiply((A - GH), (A - GH))).sum(axis=1)

                outl1_denom = alpha * pow(torch.linalg.norm((A - GH), 'fro'), 2)

                outl1_numer = outl1_numer * mu
                outl1 = outl1_numer / outl1_denom

                outl2_numer = beta * (torch.mul((C - UV), (C - UV))).sum(axis=1)

                outl2_denom = beta * pow(torch.linalg.norm((C - UV), 'fro'), 2)

                outl2_numer = outl2_numer * mu
                outl2 = outl2_numer / outl2_denom

                outl3_numer = gamma * (torch.mul((G.transpose(0, 1) - WUTrans), (G.transpose(0, 1) - WUTrans))).sum(
                    axis=0)

                outl3_denom = gamma * pow(torch.linalg.norm((G.transpose(0, 1) - WUTrans), 'fro'), 2)

                outl3_numer = outl3_numer * mu
                outl3 = outl3_numer / outl3_denom

                calc_lossValues(A, C, G, H, U, V, W, outl1, outl2, outl3)
                print('Done for outlier score')
                sys.stdout.flush()

                print('Loop {} ended: \n'.format(opti_iter))

                with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'G', 'wb') as file:
                    pickle.dump(G, file)
                with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'U', 'wb') as file:
                    pickle.dump(U, file)
                with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'W', 'wb') as file:
                    pickle.dump(W, file)
                with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'Oi1', 'wb') as file:
                    pickle.dump(outl1, file)
                with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'Oi2', 'wb') as file:
                    pickle.dump(outl2, file)
                with open(result_dir + 'pass_' + str(passNo) + '_' + str(opti_iter + 1) + '_' + 'Oi3', 'wb') as file:
                    pickle.dump(outl3, file)

            print('Pass {} Ended'.format(passNo))
