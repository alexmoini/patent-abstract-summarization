test -e ssshtest || wget -q https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest
. ssshtest

# Need to fix S3 permissions
run train_mlm python mlm_train.py --model-architecture facebook/bart-large --max-seq-length 2048 --output-dir models/bart-large/ --logs-dir logs/bart-large/ --hardware cpu --gpus 0 --num-epochs 1 --batch-size 16 --learning-rate 2e-3 --masked-lm-prob 0.20

assert_exit_code 0
