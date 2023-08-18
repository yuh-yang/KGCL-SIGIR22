## Updates
**03-07-2023:** Check our new implementation of KGCL based on the same code framework with KGIN at https://github.com/HKUDS/KGRec 

**17-01-2023:** We have rebuilt the code for KGCL to significantly improve the readability and model performance! The new version will be available soon after close check.
## KGCL

This is the Pytorch implementation for our *SIGIR'22* paper: **Knowledge Graph Contrastive Learning for Recommendation**. The CF learning part in the code is based on the open-source repository here: [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch), many thanks to the authors!

You are welcome to cite our paper:
```
@inproceedings{kgcl2022,
  author = {Yang, Yuhao and Huang, Chao and Xia, Lianghao and Li, Chenliang},
  title = {Knowledge Graph Contrastive Learning for Recommendation},
  year = {2022},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {1434–1443}
}
```

## Enviroment Requirement

`pip install -r requirements.txt`

## Dataset

We provide three processed datasets and the corresponding knowledge graphs: Yelp2018 and Amazon-book and MIND.

## An example to run KGCL

run KGCL on **Yelp2018** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command

` cd code && python main.py --dataset=yelp2018 `

` cd code && python main.py --dataset=amazon-book `

` cd code && python main.py --dataset=MIND `

## Model Variants

We also simply implement LightGCN (*SIGIR'20*) and SGL (*SIGIR'21*) for easy comparison. You can test these models implemented here by:

` cd code && python main.py --dataset=yelp2018 --model=lgn `

and

` cd code && python main.py --dataset=yelp2018 --model=sgl `

However, we still recommend to also refer to the authors' official implementation to avoid potential performance problems.
