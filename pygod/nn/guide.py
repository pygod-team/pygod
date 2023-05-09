import os
import math
import torch
import hashlib
import networkx as nx
from torch_geometric.nn import GCN
from networkx.generators.atlas import graph_atlas_g

from ..nn.encoder import GNA
from ..nn.functional import double_recon_loss


class GUIDEBase(torch.nn.Module):
    """
    Higher-order Structure based Anomaly Detection on Attributed
    Networks

    GUIDE is an anomaly detector consisting of an attribute graph
    convolutional autoencoder, and a structure graph attentive
    autoencoder (not the same as the graph attention networks). Instead
    of the adjacency matrix, node motif degree is used as input of
    structure autoencoder. The reconstruction mean square error of the
    autoencoders are defined as structure anomaly score and attribute
    anomaly score, respectively.

    Note: The calculation of node motif degree in preprocessing has
    high time complexity. It may take longer than you expect.

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    dim_a : int
        Input dimension for attribute.
    dim_s : int
        Input dimension for structure.
    hid_a : int, optional
        Hidden dimension for attribute. Default: ``64``.
    hid_s : int, optional
        Hidden dimension for structure. Default: ``4``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    **kwargs
        Other parameters for GCN.
    """
    def __init__(self,
                 dim_a,
                 dim_s,
                 hid_a=64,
                 hid_s=4,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 **kwargs):
        super(GUIDEBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.attr_encoder = GCN(in_channels=dim_a,
                                hidden_channels=hid_a,
                                num_layers=encoder_layers,
                                out_channels=dim_a,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.attr_decoder = GCN(in_channels=dim_a,
                                hidden_channels=hid_a,
                                num_layers=decoder_layers,
                                out_channels=dim_a,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.stru_encoder = GNA(in_channels=dim_s,
                                hidden_channels=hid_s,
                                num_layers=encoder_layers,
                                out_channels=dim_s,
                                dropout=dropout,
                                act=act)

        self.stru_decoder = GNA(in_channels=dim_s,
                                hidden_channels=hid_s,
                                num_layers=decoder_layers,
                                out_channels=dim_s,
                                dropout=dropout,
                                act=act)

        self.loss_func = double_recon_loss
        self.emb = ()

    def forward(self, x, s, edge_index):
        """
        Forward computation of GUIDE.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        """
        h_x = self.attr_encoder(x, edge_index)
        x_ = self.attr_decoder(h_x, edge_index)
        h_s = self.stru_encoder(s, edge_index)
        s_ = self.stru_decoder(h_s, edge_index)
        self.emb = (h_x, h_s)
        return x_, s_

    @staticmethod
    def calc_gdd(data,
                 cache_dir=None,
                 graphlet_size=4,
                 selected_motif=True):
        """
        Calculation of Node Motif Degree / Graphlet Degree
        Distribution. Part of this function is adapted
        from https://github.com/benedekrozemberczki/OrbitalFeatures.

        Parameters
        ----------
        data : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.
        cache_dir : str
            The directory for the node motif degree caching.
        graphlet_size : int, optional
            The maximum size of the graphlet. Default: 4.
        selected_motif : bool, optional
            Whether to use the selected motif or not. Default: True.

        Returns
        -------
        s : torch.Tensor
            Structure matrix (node motif degree/graphlet degree)
        """

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.pygod')

        hash_func = hashlib.sha1()
        hash_func.update(str(data).encode('utf-8'))
        file_name = 'nmd_' + str(hash_func.hexdigest()[:8]) + \
                    str(graphlet_size) + \
                    str(selected_motif)[0] + '.pt'
        file_path = os.path.join(cache_dir, file_name)
        if os.path.exists(file_path):
            s = torch.load(file_path)
        else:
            edge_index = data.edge_index
            g = nx.from_edgelist(edge_index.T.tolist())

            # create edge subsets
            edge_subsets = dict()
            subsets = [[edge[0], edge[1]] for edge in g.edges()]
            edge_subsets[2] = subsets
            unique_subsets = dict()
            for i in range(3, graphlet_size + 1):
                for subset in subsets:
                    for node in subset:
                        for neb in g.neighbors(node):
                            new_subset = subset + [neb]
                            if len(set(new_subset)) == i:
                                new_subset.sort()
                                unique_subsets[tuple(new_subset)] = 1
                subsets = [list(k) for k, v in unique_subsets.items()]
                edge_subsets[i] = subsets
                unique_subsets = dict()

            # enumerate graphs
            graphs = graph_atlas_g()
            interesting_graphs = {i: [] for i in
                                  range(2, graphlet_size + 1)}
            for graph in graphs:
                if 1 < graph.number_of_nodes() < graphlet_size + 1:
                    if nx.is_connected(graph):
                        interesting_graphs[graph.number_of_nodes()].append(
                            graph)

            # enumerate categories
            main_index = 0
            categories = dict()
            for size, graphs in interesting_graphs.items():
                categories[size] = dict()
                for index, graph in enumerate(graphs):
                    categories[size][index] = dict()
                    degrees = list(
                        set([graph.degree(node) for node in graph.nodes()]))
                    for degree in degrees:
                        categories[size][index][degree] = main_index
                        main_index += 1
            unique_motif_count = main_index

            # setup feature
            features = {node: {i: 0 for i in range(unique_motif_count)}
                        for node in g.nodes()}
            for size, node_lists in edge_subsets.items():
                graphs = interesting_graphs[size]
                for nodes in node_lists:
                    sub_gr = g.subgraph(nodes)
                    for index, graph in enumerate(graphs):
                        if nx.is_isomorphic(sub_gr, graph):
                            for node in sub_gr.nodes():
                                features[node][categories[size][index][
                                    sub_gr.degree(node)]] += 1
                            break

            motifs = [[n] + [features[n][i] for i in range(
                unique_motif_count)] for n in g.nodes()]
            motifs = torch.Tensor(motifs)
            motifs = motifs[torch.sort(motifs[:, 0]).indices, 1:]

            if selected_motif:
                # use motif selected in the original paper only
                s = torch.zeros((data.x.shape[0], 6))
                # m31
                s[:, 0] = motifs[:, 3]
                # m32
                s[:, 1] = motifs[:, 1] + motifs[:, 2]
                # m41
                s[:, 2] = motifs[:, 14]
                # m42
                s[:, 3] = motifs[:, 12] + motifs[:, 13]
                # m43
                s[:, 4] = motifs[:, 11]
                # node degree
                s[:, 5] = motifs[:, 0]
            else:
                # use graphlet degree
                s = motifs

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save(s, file_path)

        return s
