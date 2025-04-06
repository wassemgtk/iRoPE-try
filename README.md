# iRoPE Implementation on any LLM

## Overview
This project implements the iRoPE (interleaved Rotary Position Embeddings with inference-time temperature scaling) architecture on the `meta-llama/Llama-3.2-3B-Instruct` model to enhance long-context modeling (up to a simulated 10M tokens). The approach interleaves local attention (with RoPE) and global attention (with temperature scaling) to improve reasoning over long sequences, as inspired by the original iRoPE design.

## Features
- **iRoPE Architecture**: Interleaves local attention (chunked, with RoPE) and global attention (no position embeddings, with inference-time temperature scaling).
- **Long-Context Support**: Simulates a 10M-token context by processing input in chunks (16K tokens at a time).
- **Model Compatibility**: Adapts the pretrained like `Writer/Palmyra-local-1.7B`.

## Requirements
- **Hardware**: A100 or H100 
- **Libraries**:
  - `transformers`
  - `torch`
  - `accelerate`
  - `bitsandbytes`
  - `ipywidgets`
  - `huggingface_hub`

## Usage
1. **Run the Code**:
   - Open the notebook in Google Colab.
   - Set the runtime to GPU (`Runtime > Change runtime type > GPU`).
   - Run all cells to load the model, apply iRoPE, and display the UI.
2. **Interact with the UI**:
   - Enter a prompt in the input box (e.g., "Tell me a story about a futuristic city.").
   - Click the "Generate" button to process the input with a simulated 10M-token context.
   - View the output in the output box.

## Implementation Details
- **Model**: `meta-llama/Llama-3.2-3B-Instruct` / `Writer/Palmyra-local-1.7B`
- **iRoPE**:
  - Local attention: Chunked (2048 tokens) with RoPE.
  - Global attention: Full attention with inference-time temperature scaling (logarithmic by default).
- **Long-Context Simulation**: Repeats input to simulate 10M tokens, processed in 16K-token chunks.
- **Weight Transfer**: Transfers pretrained weights from the original LLaMA model to the iRoPE model to preserve performance.

## Limitations
- **No Fine-Tuning**: Uses pretrained weights directly, which may not be optimal for iRoPE. Fine-tuning on long-context tasks could improve performance.
- **Memory Constraints**: Simulates 10M tokens via chunking; actual 10M-token processing requires significant memory.
- **Simplified RoPE**: Uses a simplified RoPE implementation; production use should leverage LLaMA’s native RoPE.

## Future Work
- Fine-tune the model on long-context tasks.
- Optimize chunking to reduce boundary effects.
- Integrate LLaMA’s native RoPE implementation for better accuracy.

## Acknowledgments
- Inspired by the iRoPE architecture for infinite context modeling (@astonzhangAZ and Llama team)
- Built using the Hugging Face `transformers` library.
