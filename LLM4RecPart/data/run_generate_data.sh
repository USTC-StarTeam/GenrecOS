ITEM_JSON_PATH=../../Data/Beauty/FT_data/beauty/item_detail.json
SEQ_JSON_PATH=../../Data/Beauty/FT_data/beauty/user_seq.json
BASE_OUTPUT_PATH=../../Data/Beauty/FT_data/beauty_processed

python generate_align_data.py \
    --input_item_json ${ITEM_JSON_PATH} \
    --input_seq_json ${SEQ_JSON_PATH} \
    --output_dir ${BASE_OUTPUT_PATH}/align_data

python generate_rec_data.py \
    --input_item_json ${ITEM_JSON_PATH} \
    --input_seq_json ${SEQ_JSON_PATH} \
    --output_dir ${BASE_OUTPUT_PATH}/rec_data

python generate_cot_data.py \
    --input_item_json ${ITEM_JSON_PATH} \
    --input_seq_json ${SEQ_JSON_PATH} \
    --output_dir ${BASE_OUTPUT_PATH}/cot_data

EXPANDED_MODEL_PATH=../models/expanded_model
python generate_trie.py \
    --input_item_json ${ITEM_JSON_PATH} \
    --tokenizer_path ${EXPANDED_MODEL_PATH} \
    --output_file ${BASE_OUTPUT_PATH}/global_trie.pkl