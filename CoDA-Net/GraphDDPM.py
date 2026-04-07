import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer_denoise import GraphTransformerDenoiser
from Transformer_feature_extractor import GraphTransformerExtractor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


def create_batch(x, adj):
    data_list = []
    for i in range(x.size(0)):
        edge_index, _ = dense_to_sparse(adj[i])
        data = Data(x=x[i], edge_index=edge_index)
        data_list.append(data)
    return Batch.from_data_list(data_list)


def spectral_loss(pred, target):
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    mag_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    phase_loss = torch.mean(1 - torch.cos(torch.angle(pred_fft) - torch.angle(target_fft)))

    return mag_loss + 0.5 * phase_loss


class GraphDDPMAugmentor(nn.Module):
    def __init__(self, T, in_length, target_length, hidden_dim = 128):
        super(GraphDDPMAugmentor, self).__init__()
        self.T = T
        self.in_length = in_length
        self.target_length = target_length

        self.denoise = GraphTransformerDenoiser(in_dim=in_length, time_dim=in_length, cond_dim=2)

        self.feature_extractor = GraphTransformerExtractor(in_dim=in_length)

        self.timesteps = T

        # define beta schedule
        self.betas = self.linear_beta_schedule(timesteps=self.timesteps)
        # self.accelerator = Accelerator()
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        # out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, noise, x, e, t, c, t_index):
        shape = x.shape
        betas_t = self.extract(self.betas, t, shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, shape)
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise(noise, e, t, c) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, x, adj, c):
        device = x.device
        batch = create_batch(x, adj)
        x, edge_index = batch.x, batch.edge_index
        noise = torch.randn(x.shape, device = x.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((x.shape[0],), t, device=device)
            noise = self.p_sample(noise, x, edge_index, t_tensor, c, t)

        return noise

    def noise_pred(self, x, adj, cond0, cond1):
        x_o = x
        batch = create_batch(x, adj)
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        noise = self.q_sample(x, t)
        c = [cond0, cond1]
        predicted_noise = self.denoise(noise, edge_index, t, c)

        predicted_noise = predicted_noise.reshape_as(x_o)
        noise = noise.reshape_as(x_o)
        return F.mse_loss(predicted_noise, noise)