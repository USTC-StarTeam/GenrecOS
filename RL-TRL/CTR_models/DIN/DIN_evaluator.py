# din_evaluator.py
import torch
import os
import pickle
import numpy as np
import re
from fuxictr.features import FeatureMap
from fuxictr.utils import load_config
from . import src  # 确保 FuxiCTR 的 src 目录在 PYTHONPATH 中

class DINScorer:
    def __init__(self, config_dir, model_dir, experiment_id, data_dir, device='cuda:0'):
        self.device = device
        
        # 1. 加载 FuxiCTR 配置
        params = load_config(config_dir, experiment_id)
        params['gpu'] = int(device.split(':')[-1]) if 'cuda' in device else -1
        
        # 2. 加载数据元信息 (FeatureMap)
        feature_map_json = os.path.join(data_dir, "feature_map.json")
        if not os.path.exists(feature_map_json):
            raise FileNotFoundError(f"Feature map not found: {feature_map_json}")
            
        self.feature_map = FeatureMap(params['dataset_id'], data_dir)
        self.feature_map.load(feature_map_json, params)
        
        # 3. 加载模型权重
        model_class = getattr(src, params['model'])
        self.model = model_class(self.feature_map, **params)
        checkpoint_path = os.path.join(model_dir, experiment_id + '.model')
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
            print(f"[DINScorer] Model loaded from {checkpoint_path}")
        else:
            print(f"[DINScorer] Warning: No checkpoint found at {checkpoint_path}")
            
        self.model.to(device)
        self.model.eval()
        
        # 4. 加载动态查找表 (核心修改)
        lookup_path = os.path.join(data_dir, "1_lookup_tables.pkl")
        print(f"[DINScorer] Loading dynamic lookups from {lookup_path}...")
        
        with open(lookup_path, 'rb') as f:
            tables = pickle.load(f)
            
        # --- 4.1 基础信息 ---
        self.max_len = tables.get('max_len', 50)
        self.semantic_map = tables.get('semantic_to_item_id', {})
        self.num_categories = tables.get('num_categories', 0)
        
        # --- 4.2 动态加载 Category Maps ---
        # self.cate_maps[0] -> map_i2c1
        # self.cate_maps[1] -> map_i2c2 ...
        self.cate_maps = []
        for i in range(1, self.num_categories + 1):
            key = f'map_i2c{i}'
            if key in tables:
                # 放入 GPU tensor (或保持 numpy，推理时转 tensor，这里保持 numpy 省显存)
                self.cate_maps.append(tables[key])
            else:
                print(f"[DINScorer] Warning: Expected {key} but not found!")
                # Fallback: 全0数组
                max_iid = tables.get('num_items', 0)
                self.cate_maps.append(np.zeros(max_iid + 1, dtype=np.int32))
                
        # --- 4.3 动态加载 User Feature Dict ---
        # 结构: { 'raw_uid_str': {'enc_gender': 1, 'norm_age': 0.5, ...} }
        self.user_feat_dict = tables.get('user_feature_dict', {})
        self.user_num_cols = tables.get('user_numeric_cols', [])
        self.user_cat_cols = tables.get('user_categorical_cols', [])
        
        # 预先解析 user encoder 映射 (raw_uid -> encoded_uid)
        # FuxiCTR 的 user_id 是 categorical 特征，通常是 LabelEncoded 的
        # 我们需要知道 raw_uid 对应的 1-based index
        # 如果 pickle 里没直接存 user_id_map，我们可以尝试从 user_feat_dict 的 key 顺序或者 encoders 里恢复
        # 最稳妥的方式：查看 tables['encoders']['user_id'] (如果有)
        
        self.user_id_encoder = None
        if 'encoders' in tables and 'user_id' in tables['encoders']:
            self.user_id_encoder = tables['encoders']['user_id'] # sklearn LabelEncoder
        
        # 缓存默认 User 特征 (全0)，用于 Unknown User
        self.default_user_feats = self._build_default_user_feats()

        print(f"[DINScorer] Init Done. Categories: {self.num_categories}, Users: {len(self.user_feat_dict)}")

    def _build_default_user_feats(self):
        """构建默认的 User 特征向量（用于冷启动/未知用户）"""
        defaults = {}
        for col in self.user_cat_cols:
            defaults[f"enc_{col}"] = 0 # 假设 0 是 padding/unknown
        for col in self.user_num_cols:
            defaults[f"norm_{col}"] = 0.0
        return defaults

    def _tokens_to_item_id(self, token_str):
        return self.semantic_map.get(token_str, 0)

    def parse_completion_to_item_id(self, completion_str):
        return self._tokens_to_item_id(completion_str.strip())

    def parse_history_to_history(self, history_str):
        """
        通用解析: <a_...><b_...><c_...>
        不再依赖固定 code_depth，而是用正则切分 tokens
        """
        # 正则：匹配 <a_...> 到 <d_...> 的完整闭环，或者简单匹配 <...>
        # 你的数据是 <a_...><b_...><c_...><d_...> 拼接的
        # 假设 sid_map 的 key 也是这样拼接的，那我们只需要提取出这样的子串
        
        # 方案 A: 如果 history_str 是 "<a_1><b_2>...<a_3><b_4>..."
        # 并且 sid_map key 是 "<a_1><b_2>..."
        # 我们需要知道一个 item 由几个 <> 组成。
        
        # 尝试自动探测模式
        if not history_str: return []
        
        # 简单粗暴：正则提取所有 <...> 单元
        tokens = re.findall(r"<[^>]+>", history_str)
        if not tokens: return []
        
        # 假设 sid_map 的第一个 key 有 N 个 token
        # 我们动态计算 depth
        if not hasattr(self, 'sid_depth'):
            if len(self.semantic_map) > 0:
                sample_key = next(iter(self.semantic_map.keys()))
                self.sid_depth = sample_key.count('<') # 统计有多少个 <
            else:
                self.sid_depth = 4 # Fallback
        
        item_ids = []
        depth = self.sid_depth
        
        for i in range(0, len(tokens), depth):
            chunk = tokens[i : i + depth]
            if len(chunk) == depth:
                sid_str = "".join(chunk)
                iid = self._tokens_to_item_id(sid_str)
                if iid > 0:
                    item_ids.append(iid)
        return item_ids

    def get_user_features(self, raw_uid):
        """根据 raw_uid 获取所有用户特征"""
        raw_uid = str(raw_uid)
        
        # 1. User ID Encoding
        enc_uid = 0
        if self.user_id_encoder:
            try:
                # transform 接受 list，返回 array
                enc_uid = self.user_id_encoder.transform([raw_uid])[0] + 1
            except:
                enc_uid = 0 # Unknown
        
        # 2. Other Features
        if raw_uid in self.user_feat_dict:
            feats = self.user_feat_dict[raw_uid]
        else:
            feats = self.default_user_feats
            
        return enc_uid, feats

    def predict_batch(self, user_ids, history, completions):
        """
        user_ids: list[str] - 原始 User ID
        history: list[str] - History String
        completions: list[str] - Target SID String
        """
        batch_size = len(user_ids)
        
        # --- Pre-allocate Lists ---
        # Item Inputs
        batch_uids = []
        batch_tgts = []
        batch_hists = [] # List[List[int]]
        
        # User Feature Lists (动态)
        # 结构: {'enc_gender': [u1_val, u2_val...], ...}
        batch_user_feats = {
            f"enc_{col}": [] for col in self.user_cat_cols
        }
        for col in self.user_num_cols:
            batch_user_feats[f"norm_{col}"] = []
            
        valid_indices = []
        scores = [-999.0] * batch_size # Default low score
        
        for idx, (uid, hist_str, tgt_str) in enumerate(zip(user_ids, history, completions)):
            # 1. Target Item
            tid = self.parse_completion_to_item_id(tgt_str)
            if tid == 0:
                scores[idx] = -0.1 # Unknown Item Penalty
                continue
                
            # 2. History
            h_ids = self.parse_history_to_history(hist_str)
            # (Optional) 历史为空是否允许？DIN 通常允许，只是 Attention 为 0
            
            # 3. User Features
            enc_uid, u_feats = self.get_user_features(uid)
            
            # --- Collect ---
            batch_tgts.append(tid)
            batch_hists.append(h_ids)
            batch_uids.append(enc_uid)
            
            # User Side Feats
            for k in batch_user_feats:
                # u_feats 必须包含所有键，我们在 _build_default_user_feats 保证了这一点
                batch_user_feats[k].append(u_feats.get(k, 0))
            
            valid_indices.append(idx)
            
        if not valid_indices:
            return scores
            
        # --- Tensor Construction ---
        # 1. Target Item & Categories
        t_t = torch.tensor(batch_tgts, device=self.device).long()
        
        # 动态构造 input_dict
        input_dict = {
            'target_item_id': t_t,
            'user_id': torch.tensor(batch_uids, device=self.device).long()
        }
        
        # Dynamic Item Categories (Target)
        t_cpu = t_t.cpu().numpy()
        for i, cmap in enumerate(self.cate_maps):
            # map_i2c1, map_i2c2...
            # key in input_dict: target_cate_1, target_cate_2...
            cat_vals = cmap[t_cpu]
            input_dict[f'target_cate_{i+1}'] = torch.tensor(cat_vals, device=self.device).long()
            
        # 2. Sequence & Seq Categories
        # Padding
        padded_hists = []
        seq_lens = []
        for h in batch_hists:
            h = h[-self.max_len:] # Truncate
            l = min(len(h), self.max_len)
            padded_hists.append(h + [0] * (self.max_len - l))
            seq_lens.append(l)
            
        h_t = torch.tensor(padded_hists, device=self.device).long()
        input_dict['click_sequence'] = h_t
        input_dict['sequence_len'] = torch.tensor(seq_lens, device=self.device).long()
        
        # Dynamic Seq Categories
        h_cpu = h_t.cpu().numpy() # (B, L)
        for i, cmap in enumerate(self.cate_maps):
            # map: (NumItems,) -> h_cpu indices
            # Result: (B, L)
            # Flat map then reshape or direct map if numpy supports (it does)
            seq_cat_vals = cmap[h_cpu] 
            input_dict[f'cate_{i+1}_sequence'] = torch.tensor(seq_cat_vals, device=self.device).long()
            
        # 3. User Features
        for k, v_list in batch_user_feats.items():
            # 原始 key 是 enc_gender, norm_age...
            # FuxiCTR input key 通常是原始列名: gender, age...
            # 我们需要去掉前缀 enc_ / norm_ 吗？
            # 看你的 feature_map.json 定义。通常 FuxiCTR 的 input dict key 对应 feature_map 中的 name。
            # 如果你的 feature_map 里的 name 是 "user_active_degree"，你就得传这个 key。
            # 假设你的 dataset config 和 build script 保持一致：
            # 你的 build script 保存的是 enc_xxx。
            # 但 FuxiCTR 的 feature_map 里的 name 通常是原始列名 xxx。
            # 这里做一个 strip 处理：
            
            clean_k = k
            if k.startswith("enc_"): clean_k = k[4:]
            elif k.startswith("norm_"): clean_k = k[5:]
            
            # Numeric -> float, Categorical -> long
            if k.startswith("norm_"):
                input_dict[clean_k] = torch.tensor(v_list, device=self.device).float()
            else:
                input_dict[clean_k] = torch.tensor(v_list, device=self.device).long()

        # --- Inference ---
        with torch.no_grad():
            # FuxiCTR forward
            out = self.model(input_dict)
            pred = out['y_pred'].cpu().numpy().flatten()
            
        # Fill scores
        for i, real_idx in enumerate(valid_indices):
            scores[real_idx] = float(pred[i])
            
        return scores