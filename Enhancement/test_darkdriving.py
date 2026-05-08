import os
import os.path as osp
import logging
import argparse

import cv2
import pyiqa
import torch
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
parser.add_argument('--save_dir', type=str, default='results/darkdriving_release', help='Custom directory to save output images.')
parser.add_argument('--eval_save_name', type=str, default='pyiqa_metrics', help='Output txt name for pyiqa metrics.')
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    model = create_model(opt)
    save_dir = args.save_dir if args.save_dir else opt['path']['results_root']
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = {
        'ssim': pyiqa.create_metric('ssim', device=device),
        'psnr': pyiqa.create_metric('psnr', device=device),
        'lpips': pyiqa.create_metric('lpips', device=device),
        'musiq': pyiqa.create_metric('musiq', device=device),
        'niqe': pyiqa.create_metric('niqe', device=device),
        'hyperiqa': pyiqa.create_metric('hyperiqa', device=device),
        'cnniqa': pyiqa.create_metric('cnniqa', device=device)
    }
    metric_scores = {k: [] for k in metrics}

    print('mkdir finish')

    logger = logging.getLogger('base')


    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)

        pbar = util.ProgressBar(len(val_loader))
        psnr_rlt = {}
        psnr_rlt_avg = {}
        psnr_total_avg = 0.

        ssim_rlt = {}
        ssim_rlt_avg = {}
        ssim_total_avg = 0.

        for val_data in val_loader:
            folder = val_data['folder'][0]
            idx_d = val_data['idx']
            LQ_path = val_data['LQ_path'][0]

            if psnr_rlt.get(folder, None) is None:
                psnr_rlt[folder] = []

            if ssim_rlt.get(folder, None) is None:
                ssim_rlt[folder] = []
            model.feed_data(val_data)

            model.test()
            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])
            gt_img = util.tensor2img(visuals['GT'])

            tag = osp.join(save_dir, osp.basename(LQ_path))
            cv2.imwrite(tag, rlt_img)


            eval_lq = cv2.cvtColor(rlt_img, cv2.COLOR_BGR2RGB)
            eval_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            eval_lq = cv2.resize(eval_lq, (512, 512), interpolation=cv2.INTER_CUBIC)
            eval_gt = cv2.resize(eval_gt, (512, 512), interpolation=cv2.INTER_CUBIC)
            eval_lq = torch.from_numpy(eval_lq.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            eval_gt = torch.from_numpy(eval_gt.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            metric_scores['ssim'].append(metrics['ssim'](eval_lq, eval_gt).item())
            metric_scores['psnr'].append(metrics['psnr'](eval_lq, eval_gt).item())
            metric_scores['lpips'].append(metrics['lpips'](eval_lq, eval_gt).item())
            metric_scores['musiq'].append(metrics['musiq'](eval_lq).item())
            metric_scores['niqe'].append(metrics['niqe'](eval_lq).item())
            metric_scores['hyperiqa'].append(metrics['hyperiqa'](eval_lq).item())
            metric_scores['cnniqa'].append(metrics['cnniqa'](eval_lq).item())






            psnr = util.calculate_psnr(rlt_img, gt_img)
            psnr_rlt[folder].append(psnr)

            ssim = 0
            ssim_rlt[folder].append(ssim)

            current_psnr = sum(metric_scores['psnr']) / len(metric_scores['psnr'])
            current_ssim = sum(metric_scores['ssim']) / len(metric_scores['ssim'])
            current_lpips = sum(metric_scores['lpips']) / len(metric_scores['lpips'])
            current_niqe = sum(metric_scores['niqe']) / len(metric_scores['niqe'])
            idx_text = idx_d[0] if isinstance(idx_d, (list, tuple)) else idx_d
            pbar.update(
                'Test {} - {} | PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f} NIQE {:.4f}'.format(
                    folder, idx_text, current_psnr, current_ssim, current_lpips, current_niqe
                )
            )
        for k, v in psnr_rlt.items():
            psnr_rlt_avg[k] = sum(v) / len(v)
            psnr_total_avg += psnr_rlt_avg[k]

        for k, v in ssim_rlt.items():
            ssim_rlt_avg[k] = sum(v) / len(v)
            ssim_total_avg += ssim_rlt_avg[k]

        psnr_total_avg /= len(psnr_rlt)
        ssim_total_avg /= len(ssim_rlt)
        log_s = 'Validation PSNR: {:.4e}:'.format(psnr_total_avg)
        for k, v in psnr_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        log_s = 'Validation SSIM: {:.4e}:'.format(ssim_total_avg)
        for k, v in ssim_rlt_avg.items():
            log_s += ' {}: {:.4e}'.format(k, v)
        logger.info(log_s)

        psnr_all = 0
        psnr_count = 0
        for k, v in psnr_rlt.items():
            psnr_all += sum(v)
            psnr_count += len(v)
        psnr_all = psnr_all * 1.0 / psnr_count
        print(psnr_all)

        ssim_all = 0
        ssim_count = 0
        for k, v in ssim_rlt.items():
            ssim_all += sum(v)
            ssim_count += len(v)
        ssim_all = ssim_all * 1.0 / ssim_count
        print(ssim_all)

        pyiqa_means = {k: (sum(v) / len(v) if len(v) > 0 else float('nan')) for k, v in metric_scores.items()}
        print('==== pyiqa metrics ====')
        for metric_name, mean_value in pyiqa_means.items():
            print(f'{metric_name}: {mean_value}')
        eval_path = osp.join(save_dir, f'{args.eval_save_name}.txt')
        with open(eval_path, 'w') as f:
            f.write(f'SSIM: {pyiqa_means["ssim"]}\n')
            f.write(f'PSNR: {pyiqa_means["psnr"]}\n')
            f.write(f'LPIPS: {pyiqa_means["lpips"]}\n')
            f.write(f'MUSIQ: {pyiqa_means["musiq"]}\n')
            f.write(f'NIQE: {pyiqa_means["niqe"]}\n')
            f.write(f'HYPERIQA: {pyiqa_means["hyperiqa"]}\n')
            f.write(f'CNNIQA: {pyiqa_means["cnniqa"]}\n')
        print(f'pyiqa metrics saved to: {eval_path}')


if __name__ == '__main__':
    main()
