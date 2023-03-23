from cppimport import imp
from numpy import negative, positive
from torch_sparse.tensor import to
from model import KGCL
from random import random, sample
from shutil import make_archive
import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected
from utils import randint_choice
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import world
"""
graph shape:
[    0,     0,     0,  ..., 69714, 69715, 69715],
[    0, 31668, 31669,  ..., 69714, 31666, 69715]

values=tensor([0.0526, 0.0096, 0.0662,  ..., 0.5000, 0.1443, 0.5000])
"""


def drop_edge_random(edge_index, p):
    drop_mask = torch.empty((edge_index.size(
        1),), dtype=torch.float32, device=edge_index.device).uniform_(0, 1) < p
    x = edge_index.clone()
    x[:, drop_mask] = 0
    return x


def drop_edge_random(item2entities, p_drop, padding):
    res = dict()
    for item, es in item2entities.items():
        new_es = list()
        for e in es.tolist():
            if (random() > p_drop):
                new_es.append(e)
            else:
                new_es.append(padding)
        res[item] = torch.IntTensor(new_es).to(world.device)
    return res


def drop_edge_weighted(edge_index, edge_weights, p: float = 0.3, threshold: float = 0.7):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


class Contrast(nn.Module):
    def __init__(self, gcn_model, tau=world.kgc_temp):
        super(Contrast, self).__init__()
        self.gcn_model: KGCL = gcn_model
        self.tau = tau

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())

    def info_nce_loss_overall(self, z1, z2, z_all):
        def f(x): return torch.exp(x / self.tau)
        # batch_size
        between_sim = f(self.sim(z1, z2))
        # sim(batch_size, emb_dim || all_item, emb_dim) -> batch_size, all_item
        all_sim = f(self.sim(z1, z_all))
        # batch_size
        positive_pairs = between_sim
        # batch_size
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def get_kg_views(self):
        kg = self.gcn_model.kg_dict
        view1 = drop_edge_random(
            kg, world.kg_p_drop, self.gcn_model.num_entities)
        view2 = drop_edge_random(
            kg, world.kg_p_drop, self.gcn_model.num_entities)
        return view1, view2

    def get_ui_views_weighted(self, item_stabilities):
        graph = self.gcn_model.Graph
        n_users = self.gcn_model.num_users

        # generate mask
        item_degrees = degree(graph.indices()[0])[n_users:].tolist()
        deg_col = torch.FloatTensor(item_degrees).to(world.device)
        s_col = torch.log(deg_col)
        # degree normalization
        # deg probability of keep
        degree_weights = (s_col - s_col.min()) / (s_col.max() - s_col.min())
        degree_weights = degree_weights.where(
            degree_weights > 0.3, torch.ones_like(degree_weights) * 0.3)  # p_tau

        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        kg_weights = (item_stabilities - item_stabilities.min()) / \
            (item_stabilities.max() - item_stabilities.min())
        kg_weights = kg_weights.where(
            kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        # overall probability of keep
        weights = (1-world.ui_p_drop)/torch.mean(kg_weights)*(kg_weights)
        weights = weights.where(
            weights < 0.95, torch.ones_like(weights) * 0.95)

        item_mask = torch.bernoulli(weights).to(torch.bool)
        print(f"keep ratio: {item_mask.sum()/item_mask.size()[0]:.2f}")
        # drop
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        return g_weighted

    def item_kg_stability(self, view1, view2):
        kgv1_ro = self.gcn_model.cal_item_embedding_from_kg(view1)
        kgv2_ro = self.gcn_model.cal_item_embedding_from_kg(view2)
        sim = self.sim(kgv1_ro, kgv2_ro)
        return sim

    def ui_drop_weighted(self, item_mask):
        # item_mask: [item_num]
        item_mask = item_mask.tolist()
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        # [interaction_num]
        item_np = self.gcn_model.dataset.trainItem
        keep_idx = list()
        if world.uicontrast == "WEIGHTED-MIX":
            # overall sample rate = 0.4*0.9 = 0.36
            for i, j in enumerate(item_np.tolist()):
                if item_mask[j] and random() > 0.6:
                    keep_idx.append(i)
            # add random samples
            interaction_random_sample = sample(
                list(range(len(item_np))), int(len(item_np)*world.mix_ratio))
            keep_idx = list(set(keep_idx+interaction_random_sample))
        else:
            for i, j in enumerate(item_np.tolist()):
                if item_mask[j]:
                    keep_idx.append(i)

        print(f"finally keep ratio: {len(keep_idx)/len(item_np.tolist()):.2f}")
        keep_idx = np.array(keep_idx)
        user_np = self.gcn_model.dataset.trainUser[keep_idx]
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(
            coo.shape)).coalesce().to(world.device)
        g.requires_grad = False
        return g

    def ui_drop_random(self, p_drop):
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        keep_idx = randint_choice(len(self.gcn_model.dataset.trainUser), size=int(
            len(self.gcn_model.dataset.trainUser) * (1 - p_drop)), replace=False)
        user_np = self.gcn_model.dataset.trainUser[keep_idx]
        item_np = self.gcn_model.dataset.trainItem[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(
            coo.shape)).coalesce().to(world.device)
        g.requires_grad = False
        return g

    def get_views(self, aug_side="both"):
        # drop (epoch based)
        # kg drop -> 2 views -> view similarity for item
        if aug_side == "ui" or not world.kgc_enable:
            kgv1, kgv2 = None, None
        else:
            kgv1, kgv2 = self.get_kg_views()

        if aug_side == "kg" or world.uicontrast == "NO" or world.uicontrast == "ITEM-BI":
            uiv1, uiv2 = None, None
        else:
            if world.uicontrast == "WEIGHTED" or world.uicontrast == "WEIGHTED-MIX":
                # [item_num]
                stability = self.item_kg_stability(kgv1, kgv2).to(world.device)
                # item drop -> 2 views
                uiv1 = self.get_ui_views_weighted(stability)
                uiv2 = self.get_ui_views_weighted(stability)

            elif world.uicontrast == "RANDOM":
                uiv1 = self.ui_drop_random(world.ui_p_drop)
                uiv2 = self.ui_drop_random(world.ui_p_drop)

        contrast_views = {
            "kgv1": kgv1,
            "kgv2": kgv2,
            "uiv1": uiv1,
            "uiv2": uiv2
        }
        return contrast_views
