export CUDA_VISIBLE_DEVICES=3

python ./evaluate_single.py --dataset Beauty --model_name sasrec --checkpoint ./temp_experiment/Beauty_sasrec/sasrec_20260113_195729/best_model >> eval_test.log 2>&1