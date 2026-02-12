import torch
from flash_head.ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder


class LtxVAE:
    def __init__(
        self,
        pretrained_model_type_or_path,
        dtype = torch.bfloat16,
        device = "cuda",
    ):
        self.model = CausalVideoAutoencoder.from_pretrained(pretrained_model_type_or_path)
        self.model = self.model.eval().requires_grad_(False).to(device).to(dtype)

    # torch.Size([1, 3, 33, 512, 512]) -> torch.Size([128, 5, 16, 16])
    def encode(self, video):
        latents = self.model.encode(video, return_dict=False)[0].sample()
        out = self.normalize_latents(latents)
        return out[0]

    # torch.Size([128, 5, 16, 16]) -> torch.Size([1, 3, 33, 512, 512])
    def decode(self, zs):
        latents = zs.unsqueeze(0)
        image = self.model.decode(
            self.un_normalize_latents(latents),
            return_dict=False,
            target_shape=latents.shape,
        )[0]
        return image
    
    def normalize_latents(self, latents):
        return (
            (latents - self.model.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
            / self.model.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        )


    def un_normalize_latents(self,latents):
        return (
            latents * self.model.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
            + self.model.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        )
