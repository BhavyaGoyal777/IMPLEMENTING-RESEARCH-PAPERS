# DPO Fine-tuning on Qwen/Qwen2-0.5B-Instruct  

This repository contains a fine-tuned version of **Qwen/Qwen2-0.5B-Instruct** using **Direct Preference Optimization (DPO)**. The model was trained with **mixed precision** on **L40s GPU** using **Lightning AI Studio**.  

## Training Details  

- **Base Model**: [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)  
- **Fine-tuning Method**: DPO (Direct Preference Optimization)  
- **Batch Size**: 2  
- **OPTIMIZER**: L40s GPU on Lightning AI Studio 
- **Learning Rate**: 1e-6 (Taken from research paper)  
- **Precision**: Mixed Precision (fp16)  
- **Training Hardware**: L40s GPU on Lightning AI Studio  

## Model  

The aligned model is available on Hugging Face:  
👉 **[DPO Fine-Tuned Model](https://huggingface.co/bhavya777/dpo-sft-model?library=transformers)**  

## Performance  

Below is a visualization of the training **loss** and **accuracy**:  

![Loss and Accuracy](IMPLEMENTING-RESEARCH-PAPERS/DPO/metrics.pngpath/to/your/image.png)  

<!-- > Replace `path/to/your/image.png` with the actual path to your graph.   -->

## Usage  

To use this fine-tuned model:  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bhavya777/dpo-sft-model"
tokenizer = AutoTokenizer.from_pretrained(Qwen/Qwen2-0.5B-Instruct)
model = AutoModelForCausalLM.from_pretrained(bhavya777/dpo-sft-model)

input_text = "Your input prompt here"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
