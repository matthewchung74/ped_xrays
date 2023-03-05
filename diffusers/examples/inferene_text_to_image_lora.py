from diffusers import StableDiffusionPipeline
import torch

model_name= "stabilityai/stable-diffusion-2"
model_path = "/content/drive/MyDrive/Colab Projects/peds-xrays-hf/diffusers/examples/text_to_image/ped-xray-model-lora"
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "normal"
guidance_scale = 15
for i in range(10):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=guidance_scale).images[0]
    image.save(f"results_100/{prompt}_{guidance_scale}_{i}.png")