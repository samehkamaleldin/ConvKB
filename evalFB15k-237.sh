#!/usr/bin/env bash

python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 0 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 1 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 2 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 3 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 4 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 5 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 6 &
python eval.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit True --model_name fb15k237 --num_splits 8 --testIdx 7 &


