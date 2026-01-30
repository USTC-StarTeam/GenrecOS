export CUDA_VISIBLE_DEVICES=3

python ./evaluate_single_fast.py --dataset KuaiRec_big --model_name llamarec --batch_size 20 --checkpoint ./temp_experiment/KuaiRec_big/llama-rec_20260126_151233/best_model >> eval_test.log 2>&1