import torch
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from dataclasses import dataclass, field
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

from eval import compute_hr_at_k, compute_ndcg_at_k

@dataclass
class TestArguments:
    model_path: str = field(
        default='',
        metadata={"help": "Path to the model to be tested"}
    )
    data_path: str = field(
        default='../data/beauty_processed/cot_data/test.json',
        metadata={"help": "Path to the data for testing"}
    )
    trie_path: str = field(
        default='../data/beauty_processed/global_trie.pkl',
        metadata={"help": "Path to the trie for constrained decoding"}
    )
    # TODO: 如何选择有空闲内存的GPU?
    device: str = field(
        default='cuda',
        metadata={"help": "Device to run the model on"}
    )

@dataclass
class ThinkingArguments:
    think_max_tokens: int = field(
        default=128,
        metadata={"help": "Maximum tokens for the thinking step"}
    )
    think_temperature: float = field(
        default=1.5,
        metadata={"help": "Temperature for the thinking step"}
    )
    think_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p sampling for the thinking step"}
    )
    num_think_samples: int = field(
        default=5,
        metadata={"help": "Number of samples to generate in the thinking step"}
    )

@dataclass
class SIDArguments:
    sid_max_tokens: int = field(
        default=10,
        metadata={"help": "Maximum tokens for the SID generation step"}
    )
    sid_temperature: float = field(
        default=0.6,
        metadata={"help": "Temperature for the SID generation step"}
    )
    sid_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling for the SID generation step"}
    )
    num_sid_beams: int = field(
        default=10,
        metadata={"help": "Number of beams for the SID generation step"}
    )


def format_chat_prompt_think(user_content):
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    chat_prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    return chat_prompt

def format_chat_prompt_sid(user_content, thinking_content):
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    chat_prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n"
    )
    return chat_prompt

def extract_thinking_content(generated_text, user_content=""):
    """
    Extract the thinking content from the generated text.
    This function uses regex to find the content between <think> and </think> tags.
    """
    import re
    # 1. Remove user content safely (only first occurrence)
    if user_content:
        generated_text = generated_text.replace(user_content, "", 1)

    # 2. Define regex pattern to extract <think>...</think>
    pattern = r"<think>(.*?)(?:</think>|$)"
    
    # re.DOTALL allows '.' to match newlines
    match = re.search(pattern, generated_text, flags=re.DOTALL)
    
    return match.group(1).strip() if match else ""

import re

def extract_groundtruth_sid(generated_text):
    # 1. 定义正则
    # Part A: </think>\s* -> 匹配结束标签和随后的空白(包括换行)
    # Part B: (...)        -> 括号内的就是 Group 1，我们要提取的目标
    pattern = r"</think>\s*(<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>)"
    
    # 2. 搜索
    match = re.search(pattern, generated_text, flags=re.DOTALL)
    
    return match.group(1).strip() if match else ""
    

class Stage1Model():
    """Model wrapper for Stage 1: Thinking generation"""
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer

        self.think_max_tokens = args.think_max_tokens
        self.think_temperature = args.think_temperature
        self.think_top_p = args.think_top_p
        self.num_think_samples = args.num_think_samples

    def generate_outputs(self, input_text):
        chat_prompt_encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length, 
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=chat_prompt_encoded['input_ids'],
            attention_mask=chat_prompt_encoded['attention_mask'],
            max_new_tokens=self.think_max_tokens,
            num_beams=1,
            do_sample=True,
            temperature=self.think_temperature,
            top_p=self.think_top_p,
            return_dict_in_generate=True,
            output_scores=False,
            early_stopping=False,
            use_cache=True,
            output_hidden_states=False,
        )
        return outputs


class Stage2Model():
    """Model wrapper for Stage 2: SID generation with constraints"""
    def __init__(self, model, tokenizer, args, prefix_allowed_tokens_fn):
        self.model = model
        self.tokenizer = tokenizer

        self.sid_max_tokens = args.sid_max_tokens
        self.sid_temperature = args.sid_temperature
        self.sid_top_p = args.sid_top_p
        self.num_sid_beams = args.num_sid_beams

        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn

    def generate_outputs(self, input_text):
        chat_prompt_encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=chat_prompt_encoded['input_ids'],
            attention_mask=chat_prompt_encoded['attention_mask'],
            max_new_tokens=self.sid_max_tokens,

            num_beams=self.num_sid_beams,
            num_return_sequences=self.num_sid_beams,

            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,

            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=True,
            use_cache=True
        )
        return outputs

class JSONTestDataset(Dataset):
    def __init__(self, json_file):
        self.df = json_file
        print(f"Loaded {len(self.df)} samples from json file for testing")

        sample_num = 0  # For testing, set to 0 to use all samples
        if sample_num > 0 and sample_num < len(self.df):
            self.df = self.df.iloc[:sample_num].reset_index(drop=True)
            print(f"Limited to {sample_num} samples for this GPU")
        
        # Expected columns: ['description', 'groundtruth', 'user_id']
        required_cols = ['description', 'groundtruth']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in json file. Available: {list(self.df.columns)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'input_ids': row['description'],
            'labels': row['groundtruth'][0],
            'user_id': row.get('user_id', f'user_{idx}')
        }
    
    def get_prefix_allowed_tokens_fn(self, tokenizer, global_trie_file=None):
        """Create prefix allowed tokens function for SID constrained generation based on exact trie"""
        
        if not global_trie_file:
            raise ValueError("Global trie file path must be provided")
        
        if not os.path.exists(global_trie_file):
            raise FileNotFoundError(f"Global trie file not found: {global_trie_file}. Please run precompute_global_trie.py first.")
        
        # Load pre-computed exact trie
        import pickle
        with open(global_trie_file, 'rb') as f:
            trie_data = pickle.load(f)
        
        # Verify this is an exact trie
        trie_type = trie_data.get('trie_type', None)
        if trie_type != 'exact':
            raise ValueError(f"Expected exact trie file, but got trie_type='{trie_type}'. Please regenerate the trie file.")
        
        # Load exact trie structure
        allowed_tokens = trie_data['exact_trie']

        # Get "</think>" separator with newline (to match our prompt format)
        sep = tokenizer("</think>\n", add_special_tokens=False)["input_ids"]
        
        def find_last_sublist(lst, sub):
            """Find the last occurrence of sublist in list"""
            if not sub:
                return None
            n, m = len(lst), len(sub)
            for start in range(n - m, -1, -1):
                if lst[start:start + m] == sub:
                    return start
            return None
        
        def prefix_allowed_tokens_fn(batch_id, sentence):
            """Return allowed tokens based on current generation position using exact trie"""
            sentence = sentence.tolist()
            
            # Find "</think>" position
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                # Before "</think>", allow all tokens
                return list(tokenizer.get_vocab().values())
            
            # Calculate position after "</think>" - directly apply SID constraints
            pos_after_sep = pos + len(sep)
            generated_after_sep = sentence[pos_after_sep:]
            sid_pos = len(generated_after_sep)
            
            # Use exact trie: check what tokens are allowed at this SID position
            if sid_pos == 0:
                # First SID token position - should be <|sid_begin|>
                if 0 in allowed_tokens:
                    allowed = list(allowed_tokens[0].keys())
                    return allowed
                else:
                    # Allow all tokens if trie is not properly set up
                    return list(tokenizer.get_vocab().values())
            else:
                # Look up what's allowed based on previous SID tokens
                if sid_pos > 0 and len(generated_after_sep) >= sid_pos:
                    prev_token = generated_after_sep[sid_pos - 1]
                    prev_pos = sid_pos - 1
                    
                    if prev_pos in allowed_tokens and prev_token in allowed_tokens[prev_pos]:
                        allowed = allowed_tokens[prev_pos][prev_token]
                        return allowed
                
                # Fallback to allow all tokens if no valid continuation found
                return list(tokenizer.get_vocab().values())
        
        return prefix_allowed_tokens_fn
class TestCollator:
    """Collator for test data"""
    
    def __init__(self, tokenizer):
        # TODO: 会被调用吗?tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
    
    def __call__(self, batch):
        targets = [d["labels"] for d in batch]
        user_contents = [d["input_ids"] for d in batch]
        
        return {
            "user_contents": user_contents,
            "targets": targets
        }
    



if __name__ == "__main__":
    parser = HfArgumentParser((TestArguments, ThinkingArguments, SIDArguments))
    eval_args, think_args, sid_args = parser.parse_args_into_dataclasses()
    
    model_path = eval_args.model_path
    data_path = eval_args.data_path
    trie_path = eval_args.trie_path
    device = eval_args.device

    if not model_path:
        raise ValueError("Model path must be provided.")
    if not data_path:
        raise ValueError("Data path must be provided.")
    if not trie_path:
        raise ValueError("Trie path must be provided.")
    
    print(f"Model path provided: {model_path}")
    print(f"Data path provided: {data_path}")
    print(f"Trie path provided: {trie_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 奇怪的设置,难道没有默认值吗?
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    data = pd.read_json(data_path, lines=True)

    model.eval()

    test_dataset = JSONTestDataset(data)
    prefix_allowed_tokens_fn = test_dataset.get_prefix_allowed_tokens_fn(tokenizer, trie_path)
    
    collator = TestCollator(tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    Stage1Model_instance = Stage1Model(model, tokenizer, think_args)
    Stage2Model_instance = Stage2Model(model, tokenizer, sid_args, prefix_allowed_tokens_fn)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="CoT Testing")
        total_hr = {'HR@1': 0.0, 'HR@5': 0.0, 'HR@10': 0.0}
        total_ndcg = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0}
        total_batches = 0
        for step, batch in enumerate(progress_bar):
            user_contents = batch["user_contents"]
            targets = batch["targets"]

            # 生成思考内容部分，只有<think>和</think>之间的内容(不包含标签)
            all_think_prompts = []
            batch_mapping = []
            for sample_idx, user_content in enumerate(user_contents):
                think_prompt = format_chat_prompt_think(user_content)
                for thinking_idx in range(think_args.num_think_samples):
                    all_think_prompts.append(think_prompt)
                    batch_mapping.append((sample_idx, thinking_idx))
            think_outputs = Stage1Model_instance.generate_outputs(all_think_prompts)
            
            think_outputs_decoded = tokenizer.batch_decode(
                think_outputs["sequences"], 
                skip_special_tokens=True
            )
            
            all_thinking_contents = [[] for _ in range(len(user_contents))]
            for i, (sample_idx, thinking_idx) in enumerate(batch_mapping):
                thinking_content = extract_thinking_content(think_outputs_decoded[i], user_contents[sample_idx])
                all_thinking_contents[sample_idx].append(thinking_content)
            
            # 生成 SID 输出部分
            batch_predicted_sids = []
            for sample_idx in range(len(user_contents)):
                single_predicted_sids = []
                single_sid_scores = []
                for thinking_idx in range(think_args.num_think_samples):
                    thinking_content = all_thinking_contents[sample_idx][thinking_idx]
                    sid_prompt = format_chat_prompt_sid(user_contents[sample_idx], thinking_content)
                    sid_outputs = Stage2Model_instance.generate_outputs([sid_prompt])

                    output_ids = sid_outputs["sequences"]
                    scores = sid_outputs.get("sequences_scores", None)
                    decoded_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

                    # Process scores for this thinking sample
                    if scores is not None:
                        if hasattr(scores, 'detach'):
                            scores_batch = [float(s) for s in scores.detach().cpu().tolist()]
                        else:
                            scores_batch = [float(s) for s in scores]
                    else:
                        scores_batch = [0.0] * len(decoded_batch)

                    for decoded_text in decoded_batch:
                        predicted_sid = extract_groundtruth_sid(decoded_text)
                        single_predicted_sids.append(predicted_sid)
                    single_sid_scores.extend(scores_batch)
                sid_score_pairs = list(zip(single_predicted_sids, single_sid_scores))
                sid_score_pairs.sort(key=lambda x: x[1], reverse=True)
                sorted_sids = [pair[0] for pair in sid_score_pairs]
                seen = set()
                unique_sorted_sids = []
                for sid in sorted_sids:
                    if sid not in seen:
                        unique_sorted_sids.append(sid)
                        seen.add(sid)
                batch_predicted_sids.append(unique_sorted_sids)
            hr_k = compute_hr_at_k(batch_predicted_sids, [targets], k=[1, 5, 10])
            ndcg_k = compute_ndcg_at_k(batch_predicted_sids, [targets], k=[1, 5, 10])
            for key in total_hr.keys():
                total_hr[key] += hr_k[key]
            for key in total_ndcg.keys():
                total_ndcg[key] += ndcg_k[key]
            total_batches += 1
            if (step + 1) % 10 == 0:
                # 每10个批次打印一次平均值
                avg_hr = {key: total_hr[key] / total_batches for key in total_hr}
                avg_ndcg = {key: total_ndcg[key] / total_batches for key in total_ndcg}
                progress_bar.set_postfix({**avg_hr, **avg_ndcg})
        # 计算平均值
        avg_hr = {key: total_hr[key] / total_batches for key in total_hr}
        avg_ndcg = {key: total_ndcg[key] / total_batches for key in total_ndcg}
        print("Final Average HR:", avg_hr)
        print("Final Average NDCG:", avg_ndcg)