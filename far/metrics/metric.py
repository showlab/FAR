import lpips
import torch
from einops import rearrange
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from .fvd import FrechetVideoDistance


class VideoMetric:

    def __init__(self, metric=['fvd', 'lpips', 'mse', 'psnr', 'ssim'], device='cuda'):
        self.metric_dict = {}
        self.device = device

        if 'mse' in metric:
            self.metric_dict['mse'] = True

        if 'psnr' in metric:
            self.metric_dict['psnr'] = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3])

        if 'ssim' in metric:
            self.metric_dict['ssim'] = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(self.device)

        if 'lpips' in metric:
            self.metric_dict['lpips'] = lpips.LPIPS(net='alex', spatial=False).to(self.device)  # [0, 1]

        if 'fvd' in metric:
            self.metric_dict['fvd'] = FrechetVideoDistance().to(self.device)

    @torch.no_grad()
    def compute(self, sample, gt, context_length):
        batch_size, num_trajectory, num_frame, channel, height, width = sample.shape
        # average from per video, then select max from num_trajectory, then average from batch
        result_dict = {}

        if 'mse' in self.metric_dict:
            mse = torch.mean((sample[:, :, context_length:] - gt[:, :, context_length:])**2, dim=(3, 4, 5))
            result_dict['mse'] = float(mse.mean(-1).max(-1)[0].mean())

        if 'psnr' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous()
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous()
            psnr = self.metric_dict['psnr'](sample_, gt_)
            psnr = rearrange(psnr, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['psnr'] = float(psnr.mean(-1).max(-1)[0].mean())

        if 'ssim' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous().to(self.device)
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').contiguous().to(self.device)
            ssim = torch.zeros(sample_.shape[0])
            for start in tqdm(range(0, sample_.shape[0], 256), desc='computing ssim'):
                ssim[start:start + 256] = self.metric_dict['ssim'](sample_[start:start + 256].to(self.device), gt_[start:start + 256].to(self.device))
            ssim = rearrange(ssim, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['ssim'] = float(ssim.mean(-1).max(-1)[0].mean())

        if 'lpips' in self.metric_dict:
            sample_ = rearrange(sample[:, :, context_length:], 'b n f c h w -> (b n f) c h w').float().contiguous()
            gt_ = rearrange(gt[:, :, context_length:], 'b n f c h w -> (b n f) c h w').float().contiguous()

            lpips = torch.zeros(sample_.shape[0], 1, 1, 1).to(self.device)
            for start in tqdm(range(0, sample_.shape[0], 256), desc='computing lpips'):
                lpips[start:start + 256] = self.metric_dict['lpips'](sample_[start:start + 256].to(self.device), gt_[start:start + 256].to(self.device))
            lpips = torch.mean(lpips, dim=(1, 2, 3))
            lpips = rearrange(lpips, '(b n f) -> b n f', b=batch_size, n=num_trajectory)
            result_dict['lpips'] = float(lpips.mean(-1).min(-1)[0].mean())

        if 'fvd' in self.metric_dict and num_frame >= 10:
            sample_ = rearrange(sample, 'b n f c h w -> f b n c h w').float().contiguous()
            gt_ = rearrange(gt, 'b n f c h w -> f b n c h w').float().contiguous()
            sample_ = 2.0 * sample_ - 1
            gt_ = 2.0 * gt_ - 1

            fvd = torch.zeros(num_trajectory).to(self.device)
            for traj_idx in tqdm(range(num_trajectory), desc='computing fvd'):
                fvd[traj_idx] = self.metric_dict['fvd'].compute(sample_[:, :, traj_idx, ...], gt_[:, :, traj_idx, ...], device=self.device)
            result_dict['fvd'] = float(fvd.mean())

        return result_dict
