#!/usr/bin/env bash

python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 0 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 1 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 2 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 3 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 4 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 5 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 6 &
python eval.py --embedding_dim 50 --num_filters 500 --name WN18RR --useConstantInit False --model_name wn18rr --num_splits 8 --testIdx 7 &