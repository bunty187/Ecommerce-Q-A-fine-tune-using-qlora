import streamlit as st
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
# from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Configuration for PEFT model
# os.getenv["HUGGING_FACE_TOKEN"]=""

PEFT_MODEL = "curiousily/falcon-7b-qlora-chat-support-bot-faq"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load PEFT configuration
config = PeftConfig.from_pretrained(PEFT_MODEL)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL).to(DEVICE)

# Generation configuration
generation_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.7,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Function to generate response
def generate_response(question: str) -> str:
    prompt = f"<human>: {question}\n<assistant>:"
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start) :].strip()

# Streamlit app
st.title("AI Chatbot")
question = st.text_input("Enter your question:")
if st.button("Get Response"):
    if question:
        response = generate_response(question)
        st.write(response)
    else:
        st.write("Please enter a question.")
