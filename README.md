# Custom-BLIP-2: Training Code for Custom Swin-based BLIP-2 with Llama 3.2 LLM

This repository contains the training code for a custom BLIP-2 model that integrates a Swin Transformer for vision and Llama 3.2 (3B) for language generation. The project fine-tunes the BLIP-2 framework on a VQA dataset using custom data processing, mixed-precision training, and early stopping based on ROUGE metrics.

## Overview

- **Model Components:**
  - **Vision Backbone:** [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (`microsoft/swin-tiny-patch4-window7-224`)
  - **Language Model:** [Llama 3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
  - **Architecture:** Adaptation of the BLIP-2 framework combining a vision model and a language model with a Q-Former.

- **Dataset:**  
  The script loads a dataset from disk (using `datasets.load_from_disk`) and expects the following columns:
  - `image` (renamed to `pixel_values`)
  - `question` (renamed to `input_ids`)
  - `answer` (renamed to `labels`)
  
- **Evaluation:**  
  Uses the ROUGE metric (via the `evaluate` library) to compare generated outputs with ground truth answers.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rdinesh207/Custom-BLIP-2.git
   cd Custom-BLIP-2
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.8+ installed, then install the required packages:

   ```bash
   pip install torch torchvision transformers evaluate datasets tqdm
   ```

3. **Prepare the Dataset:**

   Update the dataset path in the training script:
   ```python
   dataset = load_from_disk("Path to dataset")
   ```

## Training

The training script performs the following steps:

- **Model Initialization:**  
  Configures the vision and language models using `Blip2VisionConfig`, `Blip2QFormerConfig`, and `Blip2Config`. The Swin Transformer and Llama model are loaded from their respective pretrained checkpoints.

- **Dataset Preparation:**  
  A custom `VQADataset` processes each sample by combining the image and text inputs using the `Blip2Processor`.

- **Training Loop:**  
  Implements mixed precision training with gradient scaling and uses early stopping based on evaluation (ROUGE scores).

- **Model Saving:**  
  The best-performing model is saved during training. For example:
  
  ```python
  model.save_pretrained("/scratch/rdinesh2/Agro_project/models/blip2_pt", from_pt=True)
  ```

## Inference

Once training is complete, use the following code snippet to load your model and perform inference:

```python
from transformers import Blip2ForConditionalGeneration, SwinModel

class CustomBlip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.vision_model = SwinModel(config.vision_config)

# Load your saved model
local_model_path = "/scratch/rdinesh2/Agro_project/models/blip2_pt"
model = CustomBlip2ForConditionalGeneration.from_pretrained(local_model_path)

# Example inference code:
# (Make sure to set up your processor as done during training)
```

For a complete example of how the fine-tuned model is used, please refer to the [HuggingFace model page](https://huggingface.co/raghavendrad60/Plant_Disease_SWIN_BLIP2_Llama3.2_3B).

## Additional Resources

- **HuggingFace Model:**  
  [Plant_Disease_SWIN_BLIP2_Llama3.2_3B](https://huggingface.co/raghavendrad60/Plant_Disease_SWIN_BLIP2_Llama3.2_3B)

- **Related GitHub Repository:**  
  This training code is part of the [Custom-BLIP-2](https://github.com/rdinesh207/Custom-BLIP-2) project.

---

Happy training and experimentation!
