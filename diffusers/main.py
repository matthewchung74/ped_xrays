import os
new_path = os.getcwd()

print(f"current path {os.getcwd()}")
print(f"to change to path {new_path}")
os.chdir(f"{new_path}/diffusers")
print(f"current path {os.getcwd()}")

import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'.'])

import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
from tqdm import tqdm
tqdm(disable=True, total=0)

st.header('stable diffusion generating xrays')

with st.form("my_form"):
    sel_col, _  =st.columns(2)
    prompt_label = sel_col.selectbox("Pick an input prompt", options=["normal", "bacteria"])

    guidance_scale = sel_col.slider("What is the gudiance", min_value=4, max_value=20)

    model_name= "stabilityai/stable-diffusion-2"
    model_path = f"{Path.cwd()}/ped-xray-model-lora"
    st.write(model_path)
    

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    pipe.unet.load_attn_procs(model_path)

    submitted = st.form_submit_button("Submit")

    image = pipe(prompt_label, num_inference_steps=30, guidance_scale=guidance_scale).images[0]

    if submitted:
       st.image(image)