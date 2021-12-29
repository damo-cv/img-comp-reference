FILE=$1
MODEL_PATH=$2
GPU=0
C=192
S=384
Z=192
LOSS=mse
export CUDA_VISIBLE_DEVICES=$GPU
python main.py --mode compress \
       --input_file ${FILE} \
       --K 1 \
       --norm GSDN \
       --num_parameter 3 \
       --channels ${C} \
       --last_channels ${S} \
       --hyper_channels ${Z} \
       --model_prefix ./ \
       --model_pretrained ${MODEL_PATH}