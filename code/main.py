from tqdm import tqdm
from contrast import Contrast
from torch import optim
from torch.optim import optimizer
import world
import utils
from world import cprint
import torch
import numpy as np
import time
import Procedure
from os.path import join
import sys
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset, kg_dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset, kg_dataset)
Recmodel = Recmodel.to(world.device)
contrast_model = Contrast(Recmodel).to(world.device)
optimizer = optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
bpr = utils.BPRLoss(Recmodel, optimizer)
weight_file = utils.getFileName()
print(f"will save to {weight_file}")
Neg_k = 1

least_loss = 1e5
best_result = 0.
stopping_step = 0

for epoch in tqdm(range(world.TRAIN_epochs), disable=True):
    start = time.time()
    # transR learning
    if world.use_trans:
        cprint("[Trans]")
        trans_loss = Procedure.TransR_train(Recmodel, optimizer)
        print(f"trans Loss: {trans_loss:.3f}")

    # joint learning part
    cprint("[Drop]")
    if world.kgc_joint:
        contrast_views = contrast_model.get_views()
    else:
        contrast_views = contrast_model.get_views("ui")
    cprint("[Joint Learning]")
    if world.kgc_joint or world.uicontrast != "NO":
        output_information = Procedure.BPR_train_contrast(dataset,
                                                            Recmodel,
                                                            bpr,
                                                            contrast_model,
                                                            contrast_views,
                                                            epoch,
                                                            optimizer,
                                                            neg_k=Neg_k)
    else:
        output_information = Procedure.BPR_train_original(dataset,
                                                            Recmodel,
                                                            bpr,
                                                            epoch,
                                                            neg_k=Neg_k)

    print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    if epoch < world.test_start_epoch:
        if epoch % 5 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset,
                            Recmodel,
                            epoch,
                            w=None,
                            multicore=world.config['multicore'])
    else:
        if epoch % world.test_verbose == 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset,
                                    Recmodel,
                                    epoch,
                                    w=None,
                                    multicore=world.config['multicore'])
            if result["recall"] > best_result:
                stopping_step = 0
                best_result = result["recall"]
                print("find a better model")
                if world.SAVE:
                    print("save...")
                    torch.save(Recmodel.state_dict(), weight_file)
            else:
                stopping_step += 1
                if stopping_step >= world.early_stop_cnt:
                    print(f"early stop triggerd at epoch {epoch}")
                    break