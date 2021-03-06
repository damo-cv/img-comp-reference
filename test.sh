FILE_PATH=$1
MODEL_PATH=$2
GPU=0
C=192
S=384
Z=192
LOSS=mse
export CUDA_VISIBLE_DEVICES=$GPU
python main.py --mode test \
       --K 1 \
       --norm GSDN \
       --num_parameter 3 \
       --channels ${C} \
       --last_channels ${S} \
       --hyper_channels ${Z} \
       --train_dir ${FILE_PATH} \
       --test_dir ${FILE_PATH} \
       --model_pretrained ${MODEL_PATH}