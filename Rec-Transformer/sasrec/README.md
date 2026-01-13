# SASRec (Self-Attentive Sequential Recommendation)

This module contains a standard Transformer-based implementation of SASRec, adapted for the Generative Recommendation framework.

Unlike the `llamarec` module which uses a Llama-2 based architecture (with RoPE, RMSNorm, SwiGLU), this module implements a more traditional Transformer architecture often used in the RecSys literature (Absolute Positional Embeddings, LayerNorm, ReLU/GELU).

## Usage

You can use this model by specifying `model_name: sasrec` in your training script arguments, provided the training script imports `SasRecForCausalLM` dynamically or staticly based on the argument.

## Structure

*   `modeling.py`: The SASRec model architecture (Embeddings + Transformer Layers + Prediction Head).
*   `configuration.py`: Configuration class (`SasRecConfig`).
*   `tokenization.py`: Utilities for handling semantic IDs (same as LlamaRec).
