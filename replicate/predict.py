# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import torch.utils.checkpoint as cp


class Predictor(BasePredictor):

    pipe = None
    device = None
    torch_dtype = None

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model running on : {device}")
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=torch_dtype,
            revision="fp16"
        ).to(device)
        pipe.enable_attention_slicing() 
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(40)
        self.pipe = pipe
        self.device = device
        self.torch_dtype = torch_dtype


    def predict(
        self,
        prompt: str = Input(description="Prompt to generate an image from stable diffusion")
    ) -> Path:
        """Pass the prompt to stable diffusion to generate an image"""
        text_input = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        text_embeddings.requires_grad_(True)
        text_embeddings.retain_grad()
        with torch.enable_grad():
            latents = torch.randn((1, self.pipe.unet.in_channels, 64, 64), device=self.device)
            latents = latents.to(self.torch_dtype) * self.pipe.scheduler.init_noise_sigma
        
        x = latents
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            model_input = self.pipe.scheduler.scale_model_input(x, t)
            print(i, end = " ")
            noise_pred = cp.checkpoint(self.unet_forward, model_input, t, text_embeddings)
            scheduler_output = self.pipe.scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample
        self.pipe.vae.eval()
        with torch.enable_grad():
            image = self.pipe.vae.decode(x / 0.18215)
            image = image.sample
        return image   
        
    def unet_forward(self, model_input, t, text_embeds):
        return self.pipe.unet(model_input, t, encoder_hidden_states=text_embeds).sample

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    predictor = Predictor()
    predictor.setup()
    image = predictor.predict("a futuristic cityscape")
    img_np = image.detach().cpu().squeeze().permute(1, 2, 0).numpy().astype(np.float32) # Squeeze if you want to remove the channel dimension
    img_np = (img_np + 1.0) / 2.0
    img_np = np.clip(img_np, 0, 1)
    plt.imsave('cog_gen.png', img_np, cmap='gray')