import sys 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
from DIN.DIN_evaluator import DINScorer

sys.path.append("Rec-Transformer") # LlamaRec 所在
from llamarec import LlamaRecConfig, LlamaRecForCausalLM 