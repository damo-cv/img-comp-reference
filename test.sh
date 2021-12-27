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
       --train_dir [training data dir] \
       --test_dir [testing data dir] \
       --model_pretrained [model_path]