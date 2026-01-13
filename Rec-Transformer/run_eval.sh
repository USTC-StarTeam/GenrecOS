export CUDA_VISIBLE_DEVICES=5

python ./evaluate_single.py --dataset Beauty --model_name llamarec --checkpoint ./temp_experiment/Beauty/llama-rec_20260112_163439/best_model >> eval.log 2>&1