# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, BaseModel, Input, Path
from typing import Any
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import torch.utils.checkpoint as cp
from pathlib import Path as LocalPath

class TokenImportance(BaseModel):
    word: str
    importance: float


class Predictor(BasePredictor):

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            torch.manual_seed(42)
        else:
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Model running on : {device}")
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=torch_dtype
        ).to(device)
        pipe.enable_attention_slicing() 
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(40)
        self.pipe = pipe
        self.device = device
        self.torch_dtype = torch_dtype


    def predict(
        self,
        prompt: str = Input(description="Prompt to generate an image from stable diffusion"),
        mode: str = Input(
            description="Mode of operation",
            choices = ["generate", "explain"]
        ),
        mask_path: Path = Input(description="Path to input mask (for 'explain' mode).", default = None)
    ) -> Any:
        if mode == "generate":
            return self.wrap_image(self.generate_image(prompt))
        elif mode == "explain":
            if mask_path is None:
                raise ValueError("mask_path is required in 'explain' mode!")
            return self.explain_mask(prompt, mask_path)
        raise ValueError(f"Unsupported mode: {mode}")
        
    def explain_mask(self, prompt: str, 
            mask_path) -> TokenImportance:
        """
        Get's individual token importance given the prompt and a mask
        """
        # image = self.generate_image(prompt)
        # self.pipe.vae.eval()
        # image_sum = image.sum()
        # image_sum.backward()
        # return self.get_token_importance()
        dummy_dict = [
            TokenImportance(word="a", importance=2.0),
            TokenImportance(word="futuristic", importance=4.0),
            TokenImportance(word="landscape", importance=6.0)
        ]
        return dummy_dict[0]

    def get_token_importance(self):
        """
        Get's individual token importance based on computed gradients
        """
        special_token_ids = set()
        print("self text input: ", self.text_input)
        print("self text embeddings : ", self.text_embeddings)

        for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
            token = getattr(self.pipe.tokenizer, attr, None)
            if token is not None:
                token_id = self.pipe.tokenizer.convert_tokens_to_ids(token)
                special_token_ids.add(token_id)

        token_importance = {}
        for token in self.pipe.tokenizer.additional_special_tokens:
            token_id = self.pipe.tokenizer.convert_tokens_to_ids(token)
            special_token_ids.add(token_id)

        special_ids_tensor = torch.tensor(list(special_token_ids), device="cpu")
        mask = ~torch.isin(self.text_input.input_ids[0], special_ids_tensor)
        importance = (self.text_embeddings.grad[0, mask] * self.text_embeddings[0, mask]).norm(dim=-1)
        final_importance = importance
        if importance.norm() > 0:
            final_importance /= importance.norm()
        if final_importance.sum() > 0:
            final_importance /= final_importance.sum()
        token_ids = self.text_input.input_ids[0, mask]
        for i, token_id in enumerate(token_ids):
            imp = final_importance[i]
            word = self.pipe.tokenizer.decode(token_id)
            token_importance[word] = imp.item()
        return token_importance
            
    def generate_image(self, prompt):
        """Pass the prompt to stable diffusion to generate an image"""
        text_input = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        self.text_input = text_input
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        text_embeddings.requires_grad_(True)
        text_embeddings.retain_grad()
        self.text_embeddings = text_embeddings
        with torch.enable_grad():
            latents = torch.randn((1, self.pipe.unet.in_channels, 64, 64), device=self.device)
            latents = latents.to(self.torch_dtype) * self.pipe.scheduler.init_noise_sigma
        
        x = latents
        print(f"Model running on : {self.device}")
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

    def wrap_image(self, image):
        """
        Takes the torch image as input and returns the image as a Path variable
        """
        import numpy as np
        import matplotlib.pyplot as plt
        img_np = image.detach().cpu().squeeze().permute(1, 2, 0).numpy().astype(np.float32) # Squeeze if you want to remove the channel dimension
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0, 1)
        output_path = "/tmp/generated_image.png"
        plt.imsave(output_path, img_np, cmap='gray')
        return Path(output_path)
        
    def unet_forward(self, model_input, t, text_embeds):
        """
        Helper function for checkpointing
        """
        return self.pipe.unet(model_input, t, encoder_hidden_states=text_embeds).sample

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    predictor = Predictor()
    predictor.setup()
    # image_path = predictor.predict("a futuristic cityscape", mode = "generate")
    # img = mpimg.imread(image_path)
    # plt.imshow(img)
    # plt.axis('off')  # Hide axes for a cleaner look
    # plt.savefig("cog_gen.png")
    token_imp = predictor.predict("a futuristic cityscape", mode = "explain", mask_path = LocalPath("fake/path.png"))
    print(token_imp)
    