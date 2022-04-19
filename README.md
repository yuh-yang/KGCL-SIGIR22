## KGCL

This is the Pytorch implementation for our *SIGIR'22* paper: **Knowledge Graph Contrastive Learning for Recommendation**. The CF learning part in the code is based on the open-source repository here: [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch), many thanks to the authors!

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