#!/bin/sh
DATA='imagenet64'
DATADIR='../data/'
SOLVER='sym12async'
KE=0.01
JF=0.01
STEPSIZE=0.25
RTOL=0.001
ATOL=0.001
SAVE=../experiments/$DATA/example_Lambda1.0_$KE$JF$SOLVER$STEPSIZE$RTOL$ATOL
LR=0.001

NUM_GPUS=2

OMP_NUM_THREADS=5 \
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
  ../train.py --data $DATA \
  --distributed \
  --batch_size 30 \
  --test_batch_size 30 \
  --datadir $DATADIR \
  --lr $LR \
  --save $SAVE \
  --solver $SOLVER  --step_size $STEPSIZE --rtol $RTOL --atol $ATOL\
  --alpha 1e-5 \
  --test_solver 'dopri5'  --test_atol 1e-5 --test_rtol 1e-5 \
  --lr 1e-3 \
  --num_epochs 200 \
  --kinetic-energy $KE \
  --jacobian-norm2 $JF \
  #--validate True \
#
