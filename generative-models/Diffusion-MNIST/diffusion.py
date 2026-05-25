import torch
from tqdm import tqdm


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=28,
        img_channels=1,
        device="cpu"
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.noise_steps
        )

    def noise_images(self, x, t):
        """
        前向扩散：把原始图片 x0 加噪成 xt。
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        noise = torch.randn_like(x)

        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        return x_t, noise

    def sample_timesteps(self, n):
        """
        为一个 batch 随机采样时间步。
        """
        return torch.randint(
            low=1,
            high=self.noise_steps,
            size=(n,),
            device=self.device
        )

    def sample(self, model, n, labels=None):
        """
        反向去噪采样：从纯噪声一步步生成图片。
        """
        model.eval()

        with torch.no_grad():
            x = torch.randn(
                n,
                self.img_channels,
                self.img_size,
                self.img_size,
                device=self.device
            )

            for i in tqdm(
                reversed(range(1, self.noise_steps)),
                desc="Sampling",
                total=self.noise_steps - 1
            ):
                t = torch.full(
                    (n,),
                    i,
                    dtype=torch.long,
                    device=self.device
                )

                if labels is None:
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (
                    1 / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / torch.sqrt(1 - alpha_hat))
                        * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )

        model.train()

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = Diffusion(
        noise_steps=1000,
        img_size=28,
        img_channels=1,
        device=device
    )

    x = torch.randn(8, 1, 28, 28).to(device)
    t = diffusion.sample_timesteps(x.shape[0])

    x_t, noise = diffusion.noise_images(x, t)

    print("原始图片形状:", x.shape)
    print("时间步形状:", t.shape)
    print("加噪图片形状:", x_t.shape)
    print("噪声形状:", noise.shape)
    print("时间步示例:", t)