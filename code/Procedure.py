'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
from contrast import Contrast
from torch.utils.data.dataloader import DataLoader
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2

def TransR_train(recommend_model, opt):
    Recmodel = recommend_model
    Recmodel.train()
    kgdataset = dataloader.KGDataset()
    kgloader = DataLoader(kgdataset,batch_size=4096,drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(world.device)
        relations = data[1].to(world.device)
        pos_tails = data[2].to(world.device)
        neg_tails = data[3].to(world.device)
        kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()


def train_contrast(recommend_model, contrast_model, contrast_views, optimizer):
    recmodel = recommend_model
    recmodel.train()
    aver_loss = 0.

    # For SGL
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    # do SGL:
        # readout
    l_kg = list()
    l_item = list()
    l_user = list()

    if world.kgc_enable:
        # item_num, emb_dim
        kgv1_readouts = recmodel.cal_item_embedding_from_kg(kgv1).split(2048)
        kgv2_readouts = recmodel.cal_item_embedding_from_kg(kgv2).split(2048)

        for kgv1_ro, kgv2_ro in zip(kgv1_readouts, kgv2_readouts):
            l_kg.append(contrast_model.semi_loss(kgv1_ro, kgv2_ro).sum())

    # if world.uic_pretrain:
    #     # user_num, emb_dim
    #     usersv1_readouts, itemsv1_readouts = recmodel.view_computer(uiv1)
    #     usersv1_readouts = usersv1_readouts.split(1024)
    #     itemsv1_readouts = itemsv1_readouts.split(1024)
    #     usersv2_readouts, itemsv2_readouts = recmodel.view_computer(uiv2)
    #     usersv2_readouts = usersv2_readouts.split(1024)
    #     itemsv2_readouts = itemsv2_readouts.split(1024)

    #     for itemsv1_ro, itemsv2_ro in zip(itemsv1_readouts, itemsv2_readouts):
    #         l_item.append(contrast_model.semi_loss(itemsv1_ro, itemsv2_ro).sum())
    #         # representations from 2 U-I views for every user
    #     for usersv1_ro, usersv2_ro in zip(usersv1_readouts, usersv2_readouts):
    #         l_user.append(contrast_model.semi_loss(usersv1_ro, usersv2_ro).sum())

    l_contrast = torch.stack(l_kg).sum()
    # l_contrast = torch.stack(l_item).sum() + torch.stack(l_user).sum()
    """
    if world.uicontrast!="NO":
        if world.uicontrast=="ITEM-BI":
            items_kg_readouts = recmodel.cal_item_embedding_from_kg(None).split(1024)
            items_cf_readouts = recmodel.computer()[1].split(1024)
            for itemsv1_ro, itemsv2_ro in zip(items_kg_readouts, items_cf_readouts):
                l_item.append(contrast_model.info_nce_loss(itemsv1_ro, itemsv2_ro).sum())

        else:
            # user_num, emb_dim
            usersv1_readouts, itemsv1_readouts = recmodel.view_computer(uiv1)
            usersv1_readouts = usersv1_readouts.split(1024)
            itemsv1_readouts = itemsv1_readouts.split(1024)
            usersv2_readouts, itemsv2_readouts = recmodel.view_computer(uiv2)
            usersv2_readouts = usersv2_readouts.split(1024)
            itemsv2_readouts = itemsv2_readouts.split(1024)

            for itemsv1_ro, itemsv2_ro in zip(itemsv1_readouts, itemsv2_readouts):
                l_item.append(contrast_model.info_nce_loss(itemsv1_ro, itemsv2_ro).sum())
                # representations from 2 U-I views for every user
            for usersv1_ro, usersv2_ro in zip(usersv1_readouts, usersv2_readouts):
                l_user.append(contrast_model.info_nce_loss(usersv1_ro, usersv2_ro).sum())
    # L = L_user + L_item + L_kg
    if world.kgcontrast:
        if world.uicontrast!="NO":
            if world.uicontrast=="ITEM-BI":
                l_contrast = torch.stack(l_kg).sum() + torch.stack(l_item).sum()
            else:
                l_contrast = torch.stack(l_kg).sum() + torch.stack(l_item).sum() + torch.stack(l_user).sum()
        else:
            l_contrast = torch.stack(l_kg).sum()
    else:
        if world.uicontrast=="ITEM-BI":
            l_contrast = torch.stack(l_item).sum()
        else:
            l_contrast = torch.stack(l_item).sum() + torch.stack(l_user).sum()
    """
    optimizer.zero_grad()
    l_contrast.backward()
    optimizer.step()

    aver_loss += l_contrast.cpu().item() / len(kgv1_readouts)
    return aver_loss


def BPR_train_contrast(dataset, recommend_model, loss_class, contrast_model :Contrast, contrast_views, epoch, optimizer, neg_k=1, w=None):
    Recmodel :model.KGCL = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    batch_size = world.config['bpr_batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)
    # with timer(name="Sample"):
    #     S = utils.UniformSample_original(dataset)
    # users = torch.Tensor(S[:, 0]).long()
    # posItems = torch.Tensor(S[:, 1]).long()
    # negItems = torch.Tensor(S[:, 2]).long()

    # users = users.to(world.device)
    # posItems = posItems.to(world.device)
    # negItems = negItems.to(world.device)
    # users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    # total_batch = len(dataset) // batch_size + 1
    total_batch = len(dataloader)
    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.
    # For SGL
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader),disable=True):
        batch_users = train_data[0].long().to(world.device)
        batch_pos = train_data[1].long().to(world.device)
        batch_neg = train_data[2].long().to(world.device)

        # main task (batch based)
        # bpr loss for a batch of users
        l_main = bpr.compute(batch_users, batch_pos, batch_neg)
        l_ssl = list()
        items = batch_pos # [B*1]

        if world.uicontrast!="NO":
            # do SGL:
                # readout
            if world.kgc_joint:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, kgv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, kgv2)
            else:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_ui(uiv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_ui(uiv2)
            # from SGL source
                # representations from 2 U-I views for every item
            items_uiv1 = itemsv1_ro[items]
            items_uiv2 = itemsv2_ro[items]
            l_item = contrast_model.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro)
            # l_item = contrast_model.grace_loss(items_uiv1, items_uiv2)
            # kgv1_ro = items_kg1_ro[items]
            # kgv2_ro = items_kg2_ro[items]
            # l_kg = contrast_model.semi_loss(kgv1_ro, kgv2_ro).sum()
                # representations from 2 U-I views for every user
            users = batch_users
            users_uiv1 = usersv1_ro[users]
            users_uiv2 = usersv2_ro[users]
            l_user = contrast_model.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro)
            # l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
            # L = L_main + L_user + L_item + L_kg + R^2
            l_ssl.extend([l_user*world.ssl_reg, l_item*world.ssl_reg])
        
        # if world.social_ssl:
        #     l_social = Recmodel.cal_social_ssl()
        #     l_ssl.append(l_social*0.01)

        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main+l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main
        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', l_all, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / (total_batch*batch_size)
    aver_loss_main = aver_loss_main / (total_batch*batch_size)
    aver_loss_ssl = aver_loss_ssl / (total_batch*batch_size)
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}-{time_info}"



def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Main"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.KGCL
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
