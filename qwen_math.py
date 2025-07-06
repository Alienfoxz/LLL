# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B")