import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
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

import os

# Configuration for PEFT model
# os.getenv["HUGGING_FACE_TOKEN"]=""

PEFT_MODEL = "curiousily/falcon-7b-qlora-chat-support-bot-faq"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
 
model = PeftModel.from_pretrained(model, PEFT_MODEL)


generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id



DEVICE = "cuda:0"


def generate_response(question: str) -> str:
    prompt = f"""
<human>: {question}
<assistant>:
""".strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start) :].strip()



st.title("Ecommerce Q/A Chatbot 🛒")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, How may I help you today?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_prompt = st.chat_input()
# answer="Clearance and final sale items are typically non-returnable and non-refundable. Please review the product description or contact our customer support team for more information.If you have any questions about our return policy, please contact our customer support team for assistance. We will be happy to assist you with the process."
if user_prompt is not None and user_prompt != "":
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))

    with st.chat_message("Human"):
        st.markdown(user_prompt)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = generate_response(user_prompt)
            st.write(response)

    st.session_state.chat_history.append(AIMessage(content=response))
