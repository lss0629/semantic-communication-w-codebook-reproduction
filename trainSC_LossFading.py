import torch.optim as optim
from net.networkCBlossFading import WITT
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
from torchvision.utils import save_image
import time
import pdb
import numpy as np
import math

parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='STL10',
                    choices=['CIFAR10', 'STL10'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='STL10',
                    choices=['CIFAR10', 'STL10'],
                    help='test dataset name')
parser.add_argument('--distortion-metric', type=str
                    , default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=32,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
parser.add_argument('--seed', type=int, default=1024,
                    help='random seed')
parser.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader worker processes (0 is safest on Windows)')
parser.add_argument('--lambda_loss', type=float, default='0.01',
                    help='lambda in the loss function')
parser.add_argument('--SCsize', type=int, default=32,
                    choices=[10, 16, 32, 64],
                    help='SC size')
parser.add_argument('--smoke-test', action='store_true',
                    help='run exactly one training batch without saving checkpoints')
parser.add_argument('--smoke-batch-size', type=int, default=1,
                    help='batch size used only with --smoke-test')
parser.add_argument('--benchmark-one-epoch', action='store_true',
                    help='run one training epoch with batch size 4 and no checkpoint/evaluation')
parser.add_argument('--pilot-training', action='store_true',
                    help='run the bounded pilot path with per-epoch evaluation and pilot checkpoints')
parser.add_argument('--resume-pilot', type=str, default=None,
                    help='resume pilot codec weights from a checkpoint; optimizer state is not restored')
parser.add_argument('--eval-final-only', action='store_true',
                    help='evaluate only after the final requested pilot epoch')
parser.add_argument('--epochs', type=int, default=500000,
                    help='number of training epochs (original default: 500000)')
parser.add_argument('--batch-size', type=int, default=12,
                    help='STL10 training batch size (original default: 12)')
args = parser.parse_args()
if args.resume_pilot:
    args.pilot_training = True
if sum((args.smoke_test, args.benchmark_one_epoch, args.pilot_training)) > 1:
    parser.error('--smoke-test, --benchmark-one-epoch, and --pilot-training are mutually exclusive')
if args.eval_final_only and not args.pilot_training:
    parser.error('--eval-final-only requires --pilot-training or --resume-pilot')
if args.epochs <= 0:
    parser.error('--epochs must be positive')
if args.batch_size <= 0:
    parser.error('--batch-size must be positive')
if args.smoke_test or args.benchmark_one_epoch:
    if args.smoke_batch_size <= 0:
        parser.error('--smoke-batch-size must be positive')
    args.training = True
    args.model = 'WITT_W/O'
    args.channel_type = 'awgn'
    args.C = 4
    args.SCsize = 32
    args.distortion_metric = 'MSE'
    args.multiple_snr = '10'
    args.lambda_loss = 0.01
    args.num_workers = 0
if args.pilot_training:
    args.training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()

        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(GoogLeNet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x): 
        x = self.block1(x) 
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))

        x = x.view(x.shape[0], -1)  
        x = self.classifier(x)
        return x

class config():
    seed = 1024  # random seed
    pass_channel = True
    CUDA = torch.cuda.is_available()
    device = device
    num_workers = args.num_workers
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = windows_safe_timestamp()
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    # learning_rate = 0.0005

    # tot_epoch = 10000000
    tot_epoch = args.epochs

    if args.trainset == 'CIFAR10':
        save_model_freq = 50  # save model epoch
        image_dims = (3, 32, 32)
        train_data_dir = "./media/Dataset/CIFAR10/"
        test_data_dir = "./media/Dataset/CIFAR10/"
        # batch_size = 128 
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    
    elif args.trainset == 'STL10':
        alpha = 100  # parameter in loss function, need to be adjust

        # CBsize = 32 # 10, 16, 32, 64
        save_model_freq = 10  # save model epoch and results

        image_dims = (3, 256, 256)
        # image_dims = (3, 96, 96)
        train_data_dir = "./media/Dataset/CIFAR10/"
        test_data_dir = "./media/Dataset/CIFAR10/"
        # batch_size = 128 
        if args.smoke_test:
            batch_size = args.smoke_batch_size
        elif args.benchmark_one_epoch:
            batch_size = 4
        else:
            batch_size = args.batch_size
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )



CalcuSSIM = None

def load_weights(model_path):
    pretrained = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(pretrained, strict=True)
    del pretrained

def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


def run_smoke_test(args, H_fading_all):
    """Run one real training batch while leaving the full-training path untouched."""
    net.train()
    classifier.train()
    smoke = {'stage': 'setup'}
    hooks = []

    def encoder_pre_hook(module, inputs):
        smoke['stage'] = 'encoder forward'

    def encoder_hook(module, inputs, output):
        latent, mu, std = output
        smoke['latent_shape'] = tuple(latent.shape)
        smoke['mu_shape'] = tuple(mu.shape)
        smoke['std_shape'] = tuple(std.shape)
        smoke['stage'] = 'proposed loss calculation'

    def channel_pre_hook(module, inputs):
        smoke['stage'] = 'AWGN channel'

    def channel_hook(module, inputs, output):
        smoke['channel_output_shape'] = tuple(output.shape)

    def decoder_pre_hook(module, inputs):
        smoke['channel_output_shape'] = tuple(inputs[0].shape)
        smoke['stage'] = 'decoder forward'

    def decoder_hook(module, inputs, output):
        smoke['decoder_output_shape'] = tuple(output.shape)
        smoke['stage'] = 'distortion loss'

    hooks.append(net.encoder.register_forward_pre_hook(encoder_pre_hook))
    hooks.append(net.encoder.register_forward_hook(encoder_hook))
    hooks.append(net.channel.register_forward_pre_hook(channel_pre_hook))
    hooks.append(net.channel.register_forward_hook(channel_hook))
    hooks.append(net.decoder.register_forward_pre_hook(decoder_pre_hook))
    hooks.append(net.decoder.register_forward_hook(decoder_hook))

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    try:
        smoke['stage'] = 'data loading'
        input_image, label = next(iter(train_loader))
        input_image = input_image.to(device)
        label = label.to(device)
        smoke['input_shape'] = tuple(input_image.shape)

        smoke['stage'] = 'codeword search'
        code_assist = input_image.clone()
        code_index = []
        for image_ID in range(input_image.size()[0]):
            code_index_local = 0
            mse_ini = 10 ** 8
            for assist_ID in range(codebook.size()[0]):
                mse_local = MSE_loss(input_image[image_ID], codebook[assist_ID])
                if mse_local < mse_ini:
                    code_assist[image_ID] = codebook[assist_ID].clone()
                    code_index_local = assist_ID
                    mse_ini = mse_local
            code_index.append(code_index_local)
        code_index = torch.from_numpy(np.array(code_index)).to(device)
        smoke['selected_codeword_shape'] = tuple(code_assist.shape)
        smoke['selected_codeword_index'] = code_index.detach().cpu().tolist()

        smoke['stage'] = 'residual calculation'
        residual = torch.sub(input_image, code_assist)
        smoke['residual_shape'] = tuple(residual.shape)
        del residual

        H_fading = H_fading_all[0]
        recon_image, CBR, actual_snr, mse, loss_G, loss_P = net(
            input_image, code_assist, code_index, H_fading)
        smoke['reconstruction_shape'] = tuple(recon_image.shape)

        # Preserve the classifier forward performed by the normal training path.
        smoke['stage'] = 'classifier forward'
        downsampled_image = downsampling(recon_image, 96)
        out_class = classifier(downsampled_image)
        loss_C = CE_loss(out_class, label)

        smoke['stage'] = 'total loss'
        loss = (loss_G + args.lambda_loss * loss_P).clone()
        smoke['losses_finite'] = bool(
            torch.isfinite(loss_G).all().item()
            and torch.isfinite(loss_P).all().item()
            and torch.isfinite(loss).all().item()
            and torch.isfinite(loss_C).all().item())

        smoke['stage'] = 'backward'
        optimizer.zero_grad()
        loss.backward()
        smoke['backward_success'] = True

        smoke['stage'] = 'optimizer.step'
        optimizer.step()
        smoke['optimizer_step_success'] = True
        smoke['output_has_nan'] = bool(torch.isnan(recon_image).any().item())
        smoke['output_has_inf'] = bool(torch.isinf(recon_image).any().item())

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0
        runtime = time.perf_counter() - start_time

        print('[Smoke Test Result]')
        print('configuration: model={} channel={} C={} SCsize={} distortion={} multiple_snr={} lambda_loss={} batch_size={}'.format(
            args.model, args.channel_type, args.C, args.SCsize, args.distortion_metric,
            args.multiple_snr, args.lambda_loss, args.smoke_batch_size))
        print('input_image_shape:', smoke['input_shape'])
        print('selected_codeword_shape:', smoke['selected_codeword_shape'])
        print('selected_codeword_index:', smoke['selected_codeword_index'])
        print('residual_shape:', smoke['residual_shape'])
        print('encoder_latent_shape:', smoke['latent_shape'])
        print('mu_shape:', smoke['mu_shape'])
        print('std_shape:', smoke['std_shape'])
        print('channel_output_shape:', smoke['channel_output_shape'])
        print('decoder_output_shape:', smoke['decoder_output_shape'])
        print('final_reconstruction_shape:', smoke['reconstruction_shape'])
        print('actual_snr:', float(actual_snr))
        print('distortion_loss:', float(loss_G.detach().item()))
        print('proposed_loss:', float(loss_P.detach().item()))
        print('total_loss:', float(loss.detach().item()))
        print('losses_finite:', smoke['losses_finite'])
        print('backward_success:', smoke['backward_success'])
        print('optimizer_step_success:', smoke['optimizer_step_success'])
        print('output_has_nan:', smoke['output_has_nan'])
        print('output_has_inf:', smoke['output_has_inf'])
        print('gpu_peak_allocated_mib: {:.2f}'.format(peak_allocated))
        print('gpu_peak_reserved_mib: {:.2f}'.format(peak_reserved))
        print('runtime_seconds: {:.3f}'.format(runtime))
        return True
    except torch.cuda.OutOfMemoryError:
        if device.type == 'cuda':
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0
        runtime = time.perf_counter() - start_time
        print('[Smoke Test OOM]')
        print('stage:', smoke['stage'])
        print('gpu_peak_allocated_mib: {:.2f}'.format(peak_allocated))
        print('gpu_peak_reserved_mib: {:.2f}'.format(peak_reserved))
        print('runtime_seconds: {:.3f}'.format(runtime))
        return False
    finally:
        for hook in hooks:
            hook.remove()


def train_one_epoch(args, lambda_loss_local, H_fading_all):
    error_time = 0
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs, accs = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs, accs]
    benchmark_stats = None
    if args.benchmark_one_epoch or args.pilot_training:
        benchmark_stats = {
            'batches': 0,
            'processed_images': 0,
            'first_loss': None,
            'final_loss': None,
            'distortion_loss_sum': 0.0,
            'proposed_loss_sum': 0.0,
            'total_loss_sum': 0.0,
            'has_nan': False,
            'has_inf': False,
            'backward_success': True,
            'optimizer_step_success': True,
        }
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        benchmark_start = time.perf_counter()
    global global_step
    if args.trainset == 'CIFAR10' or args.trainset == 'STL10':
        for batch_idx, (input, label) in enumerate(train_loader): 

            H_id = int(epoch * batch_idx) % 19999
            H_fading = H_fading_all[H_id]

            start_time = time.time()

            global_step += 1
            input = input.to(device)
            label = label.to(device)

            # search for the codeword 
            code_assist = input.clone() 
            code_index = []
            for image_ID in range(input.size()[0]):
                code_index_local = 0
                mse_ini = 10 ** 8
                for assist_ID in range(codebook.size()[0]):
                    mse_local = MSE_loss(input[image_ID], codebook[assist_ID])
                    if mse_local < mse_ini:
                        code_assist[image_ID] = codebook[assist_ID].clone()
                        code_index_local = assist_ID
                        mse_ini = mse_local
                # print('code_index_local', code_index_local)
                code_index.append(code_index_local)

            code_index = torch.from_numpy(np.array(code_index)).to(device)
            recon_image, CBR, SNR, mse, loss_G, loss_P = net(input, code_assist, code_index, H_fading)  # loss_G is the loss for generating image

            # loss_G = loss_G + config.alpha * loss_P
            loss = (loss_G + lambda_loss_local * loss_P).clone()

            if math.isnan(loss.item()):
                print('Loss error! Please choose another lambda!')
                pdb.set_trace()
            
            if loss.item() >= 0:
                pass
            else:
                print('loss G:', loss_G.item())  
                print('loss P:', loss_P.item())
                # lambda_loss_local = lambda_loss_local / 2
                print('lambda_loss:', lambda_loss_local)
                loss = loss_G.clone()
                # pdb.set_trace()

            out_class = None
            if args.trainset == 'STL10':
                downsampled_image  = downsampling(recon_image, 96)  # from 256 * 256 to 96 * 96
                out_class = classifier(downsampled_image)
            else:
                out_class = classifier(recon_image)
            loss_C = CE_loss(out_class, label)  # loss_G is the loss for classification

            _, pred = out_class.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / input.shape[0] * 100

            if epoch >= 40 and epoch % 20 == 0:
                optimizer_classifier.zero_grad()
                loss_C.backward() 
                optimizer_classifier.step()

            else:
                # update the coder NNs
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                elapsed.update(time.time() - start_time)
                losses.update(loss.item())
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                    accs.update(acc)
                else:
                    psnrs.update(100)
                    msssims.update(100)
                    accs.update(100)

                if not (args.benchmark_one_epoch or args.pilot_training) and batch_idx % 50 == 0 and epoch % 5 == 0:
                    # save image                      
                    recon_image = downsampling(recon_image, 512)
                    # recon_image0 = downsampling(input, 512)
                    for iii in range(input.size()[0]):
                        # save_image(recon_image0[iii], ('./image_raw/img%d_epoch%d_batch%d_snr%d.png' % (iii, epoch, batch_idx, snr)))  # save the raw images
                        if args.channel_type == 'awgn':
                            save_image(recon_image[iii], ('./image_recover_SC_loss/img%d_epoch%d_batch%d_snr%d.png' % (iii, epoch, batch_idx, snr)))
                        else:
                            save_image(recon_image[iii], ('./image_recover_SC_loss_Fading/img%d_epoch%d_batch%d_snr%d.png' % (iii, epoch, batch_idx, snr)))     

                if (global_step % config.print_step) == 0:
                    process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Acc {accs.val:.3f} ({accs.avg:.3f})', 
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
                    for i in metrics:
                        i.clear()
                    
                    # add the training and validating results
                    val_Acc_all.append(accs.val)
                    train_Acc_all.append(accs.avg)
                    val_PSNR_all.append(psnrs.val)
                    train_PSNR_all.append(psnrs.avg)
                    val_SSIM_all.append(msssims.val)
                    train_SSIM_all.append(msssims.avg)

            if benchmark_stats is not None:
                distortion_value = float(loss_G.detach().item())
                proposed_value = float(loss_P.detach().item())
                total_value = float(loss.detach().item())
                if benchmark_stats['first_loss'] is None:
                    benchmark_stats['first_loss'] = total_value
                benchmark_stats['final_loss'] = total_value
                benchmark_stats['distortion_loss_sum'] += distortion_value
                benchmark_stats['proposed_loss_sum'] += proposed_value
                benchmark_stats['total_loss_sum'] += total_value
                benchmark_stats['batches'] += 1
                benchmark_stats['processed_images'] += input.size(0)
                benchmark_stats['has_nan'] = (
                    benchmark_stats['has_nan']
                    or math.isnan(distortion_value)
                    or math.isnan(proposed_value)
                    or math.isnan(total_value)
                    or bool(torch.isnan(recon_image).any().item()))
                benchmark_stats['has_inf'] = (
                    benchmark_stats['has_inf']
                    or math.isinf(distortion_value)
                    or math.isinf(proposed_value)
                    or math.isinf(total_value)
                    or bool(torch.isinf(recon_image).any().item()))
                if args.benchmark_one_epoch and (
                        benchmark_stats['batches'] % 100 == 0
                        or benchmark_stats['batches'] == len(train_loader)):
                    benchmark_elapsed = time.perf_counter() - benchmark_start
                    if device.type == 'cuda':
                        current_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    else:
                        current_peak = 0.0
                    print('benchmark batch: {}/{} | images: {} | elapsed: {:.3f} seconds | peak allocated: {:.2f} MiB'.format(
                        benchmark_stats['batches'], len(train_loader),
                        benchmark_stats['processed_images'], benchmark_elapsed, current_peak), flush=True)

    for i in metrics:
        i.clear()
    if benchmark_stats is not None:
        if args.benchmark_one_epoch:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                benchmark_stats['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                benchmark_stats['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            else:
                benchmark_stats['peak_allocated_mib'] = 0.0
                benchmark_stats['peak_reserved_mib'] = 0.0
            benchmark_stats['epoch_runtime'] = time.perf_counter() - benchmark_start
            benchmark_stats['average_batch_runtime'] = (
                benchmark_stats['epoch_runtime'] / benchmark_stats['batches'])
        benchmark_stats['mean_distortion_loss'] = (
            benchmark_stats['distortion_loss_sum'] / benchmark_stats['batches'])
        benchmark_stats['mean_proposed_loss'] = (
            benchmark_stats['proposed_loss_sum'] / benchmark_stats['batches'])
        benchmark_stats['mean_total_loss'] = (
            benchmark_stats['total_loss_sum'] / benchmark_stats['batches'])
        return lambda_loss_local, benchmark_stats
    return lambda_loss_local

def test(H_fading_all):
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs, accs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs, accs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_acc = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        if args.pilot_training:
            pilot_test_start = time.perf_counter()
        with torch.no_grad():
            if args.trainset == 'CIFAR10' or args.trainset == 'STL10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.to(device)
                    label = label.to(device)

                    H_id = int(epoch * batch_idx) % 19999
                    H_fading = H_fading_all[H_id]  

                    # search for the codeword
                    code_assist = input.clone() 
                    code_index = [] 
                    for image_ID in range(input.size()[0]):
                        code_index_local = 0
                        mse_ini = 10 ** 8
                        for assist_ID in range(codebook.size()[0]):
                            mse_local = MSE_loss(input[image_ID], codebook[assist_ID])
                            if mse_local < mse_ini:
                                code_assist[image_ID] = codebook[assist_ID].clone()
                                code_index_local = assist_ID
                                mse_ini = mse_local
                        code_index.append(code_index_local)

                    code_index = torch.from_numpy(np.array(code_index)).to(device)
                    recon_image, CBR, SNR, mse, loss_G, loss_P = net(input, code_assist, code_index, H_fading)  # loss_G is the loss for generating image

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)

                    recon_image_down  = downsampling(recon_image, 96)  # from 256 * 256 to 96 * 96
                    out_class = classifier(recon_image_down)

                    _, pred = out_class.max(1)
                    num_correct = (pred == label).sum().item()
                    acc = num_correct / input.shape[0] * 100


                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                        accs.update(acc)
                    else:
                        psnrs.update(100)
                        msssims.update(100)
                        accs.update(100)

                    if args.pilot_training and ((batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader)):
                        print('pilot test batch: {}/{} | images: {} | elapsed: {:.3f} seconds'.format(
                            batch_idx + 1, len(test_loader),
                            min((batch_idx + 1) * input.shape[0], len(test_loader.dataset)),
                            time.perf_counter() - pilot_test_start), flush=True)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Acc {accs.val:.3f} ({accs.avg:.3f})', 
                        f'Lr {cur_lr}',
                    ]))
                    # logger.info(log)  # print the print of the info

        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        results_acc[i] = accs.avg
        for t in metrics:
            t.clear()
        
        # add the testing results
        test_Acc_all.append(accs.avg)
        test_PSNR_all.append(psnrs.avg)
        test_SSIM_all.append(msssims.avg)

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("Acc: {}" .format(results_acc.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")
    return {
        'snr': results_snr.tolist(),
        'cbr': results_cbr.tolist(),
        'psnr': results_psnr.tolist(),
        'msssim': results_msssim.tolist(),
        'accuracy': results_acc.tolist(),
    }


def global_func():
    global train_PSNR_all
    global train_Acc_all
    global train_SSIM_all
    global val_PSNR_all
    global val_Acc_all
    global val_SSIM_all
    global test_PSNR_all
    global test_Acc_all
    global test_SSIM_all

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=args.seed)

    if args.trainset == 'CIFAR10':
        CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(device)
    else:
        CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(device)

    for output_dir in (
            './results_data', './results_data/results_SC_loss',
            './results_data/results_SC_loss_Fading',
            './saved_model/awgn/STL10', './saved_model/rayleigh/STL10',
            './saved_model/pilot',
            './image_recover_SC_loss', './image_recover_SC_loss_Fading'):
        makedirs(output_dir)

    snr_ini = int(args.multiple_snr.split(",")[0])

    net = WITT(args, config)

    snr = int(args.multiple_snr.split(",")[0])

    # load the codebook
    codebook_np = np.load('./results_data/SC_size' + str(args.SCsize) + '.npy')
    codebook = torch.from_numpy(codebook_np).to(device)
    codebook = codebook.view(args.SCsize, 3, 256, 256)

    if args.channel_type == 'awgn':
        pre_model_exist = False
    else:
        # use the pre-trained model for fading channel
        model_path = "./saved_model/awgn/STL10/SC_loss_snr" + str(snr_ini) + "_C" + str(args.C) + ".model"
        pre_model_exist = os.path.isfile(model_path)  # if the pre-trained model exists
        if pre_model_exist:
            load_weights(model_path)
            print('*' * 50)
            print('load model parameters ...')

    CE_loss = nn.CrossEntropyLoss()
    MSE_loss = nn.MSELoss()
    classifier = GoogLeNet(3, 10)  
    classifier.load_state_dict(torch.load('google_net.pkl', map_location=device, weights_only=True))
    classifier.to(device)

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=0.0001)  # fine-tune the classifier, the learning rate should be very small

    train_PSNR_all = []
    train_Acc_all = []
    train_SSIM_all = []
    val_PSNR_all = []
    val_Acc_all = []
    val_SSIM_all = []
    test_PSNR_all = []
    test_Acc_all = []
    test_SSIM_all = []
    global_func()

    net = net.to(device)
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)

    if args.channel_type == 'awgn':
        H_fading_all = np.zeros(20000)
    else:
        H_fading_all = np.sqrt(2 / np.pi) *  np.random.rayleigh(1, 20000)  # generate the fading coefficient
        H_fading_all = 10 * np.log10(H_fading_all + 10 ** (-10))

    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.smoke_test:
        epoch = 0
        cur_lr = 0.01
        optimizer = optim.Adam(model_params, lr=cur_lr)
        if not run_smoke_test(args, H_fading_all):
            raise SystemExit(1)
    elif args.benchmark_one_epoch:
        epoch = 0
        cur_lr = 0.01
        optimizer = optim.Adam(model_params, lr=cur_lr)
        _, benchmark_stats = train_one_epoch(args, args.lambda_loss, H_fading_all)
        print('[One Epoch Benchmark Result]')
        print('configuration: model={} channel={} C={} SCsize={} distortion={} multiple_snr={} lambda_loss={} batch_size={}'.format(
            args.model, args.channel_type, args.C, args.SCsize, args.distortion_metric,
            args.multiple_snr, args.lambda_loss, config.batch_size))
        print('total_batches:', benchmark_stats['batches'])
        print('processed_images:', benchmark_stats['processed_images'])
        print('first_batch_total_loss:', benchmark_stats['first_loss'])
        print('final_batch_total_loss:', benchmark_stats['final_loss'])
        print('epoch_mean_distortion_loss:', benchmark_stats['mean_distortion_loss'])
        print('epoch_mean_proposed_loss:', benchmark_stats['mean_proposed_loss'])
        print('epoch_mean_total_loss:', benchmark_stats['mean_total_loss'])
        print('learning_rate:', cur_lr)
        print('epoch_runtime_seconds: {:.3f}'.format(benchmark_stats['epoch_runtime']))
        print('average_batch_runtime_seconds: {:.6f}'.format(benchmark_stats['average_batch_runtime']))
        print('gpu_peak_allocated_mib: {:.2f}'.format(benchmark_stats['peak_allocated_mib']))
        print('gpu_peak_reserved_mib: {:.2f}'.format(benchmark_stats['peak_reserved_mib']))
        print('has_nan:', benchmark_stats['has_nan'])
        print('has_inf:', benchmark_stats['has_inf'])
        print('backward_success:', benchmark_stats['backward_success'])
        print('optimizer_step_success:', benchmark_stats['optimizer_step_success'])
    elif args.pilot_training:
        lambda_loss = args.lambda_loss
        pilot_history = []
        pilot_checkpoint = './saved_model/pilot/latest.pt'
        pilot_start_epoch = steps_epoch

        if args.resume_pilot:
            resume_checkpoint = torch.load(args.resume_pilot, map_location='cpu', weights_only=True)
            if 'model_state_dict' not in resume_checkpoint or 'epoch' not in resume_checkpoint:
                raise RuntimeError('Pilot resume checkpoint must contain model_state_dict and epoch.')
            net.load_state_dict(resume_checkpoint['model_state_dict'], strict=True)
            pilot_start_epoch = int(resume_checkpoint['epoch'])
            pilot_history = list(resume_checkpoint.get('pilot_history', []))
            global_step = pilot_start_epoch * len(train_loader)
            print('[Pilot Resume] checkpoint:', args.resume_pilot, flush=True)
            print('[Pilot Resume] codec model_state_dict loaded with strict=True', flush=True)
            print('[Pilot Resume] checkpoint epoch={} | next internal epoch={} | next displayed epoch={}'.format(
                pilot_start_epoch, pilot_start_epoch, pilot_start_epoch + 1), flush=True)
            print('[Pilot Resume] optimizer_state_dict intentionally not loaded; Adam is recreated each epoch.', flush=True)
            if 'classifier_state_dict' not in resume_checkpoint:
                classifier_warning = (
                    'Epoch 10 checkpoint has no classifier_state_dict. The classifier is initialized from '
                    'google_net.pkl; BatchNorm running statistics changed during Epochs 1-10 cannot be restored.'
                )
                print('[Pilot Resume Warning]', classifier_warning, flush=True)
                logger.warning(classifier_warning)
            else:
                classifier.load_state_dict(resume_checkpoint['classifier_state_dict'], strict=True)
                print('[Pilot Resume] classifier_state_dict loaded with strict=True', flush=True)
            del resume_checkpoint

        if pilot_start_epoch >= config.tot_epoch:
            raise RuntimeError('Resume epoch {} must be less than --epochs {}.'.format(
                pilot_start_epoch, config.tot_epoch))

        pilot_total_start = time.perf_counter()

        for epoch in range(pilot_start_epoch, config.tot_epoch):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            if epoch < 200:
                cur_lr = 0.01
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 400:
                cur_lr = 0.005
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 550:
                cur_lr = 0.002
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 650:
                cur_lr = 0.001
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 750:
                cur_lr = 0.0005
                optimizer = optim.Adam(model_params, lr=cur_lr)
            else:
                cur_lr = 0.0001
                optimizer = optim.Adam(model_params, lr=cur_lr)

            pilot_train_start = time.perf_counter()
            try:
                lambda_loss, train_stats = train_one_epoch(args, lambda_loss, H_fading_all)
            except torch.cuda.OutOfMemoryError:
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                print('Pilot stopped: CUDA OOM during training at epoch {} | peak allocated {:.2f} MiB | peak reserved {:.2f} MiB'.format(
                    epoch + 1, peak_allocated, peak_reserved))
                raise SystemExit(1)
            if train_stats['has_nan'] or train_stats['has_inf']:
                print('Pilot stopped: NaN/Inf detected in training at epoch {}.'.format(epoch + 1))
                raise SystemExit(1)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_runtime = time.perf_counter() - pilot_train_start

            should_evaluate = (not args.eval_final_only) or (epoch + 1 == config.tot_epoch)
            test_psnr = None
            test_msssim = None
            test_accuracy = None
            evaluation_runtime = 0.0
            if should_evaluate:
                pilot_evaluation_start = time.perf_counter()
                try:
                    test_stats = test(H_fading_all)
                except torch.cuda.OutOfMemoryError:
                    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                    print('Pilot stopped: CUDA OOM during evaluation at epoch {} | peak allocated {:.2f} MiB | peak reserved {:.2f} MiB'.format(
                        epoch + 1, peak_allocated, peak_reserved))
                    raise SystemExit(1)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                evaluation_runtime = time.perf_counter() - pilot_evaluation_start
                test_psnr = float(test_stats['psnr'][0])
                test_msssim = float(test_stats['msssim'][0])
                test_accuracy = float(test_stats['accuracy'][0])
                if not all(math.isfinite(value) for value in (test_psnr, test_msssim, test_accuracy)):
                    print('Pilot stopped: NaN/Inf detected in evaluation at epoch {}.'.format(epoch + 1))
                    raise SystemExit(1)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            else:
                peak_allocated = 0.0
                peak_reserved = 0.0
            epoch_runtime = train_runtime + evaluation_runtime

            epoch_result = {
                'epoch': epoch + 1,
                'learning_rate': cur_lr,
                'train_distortion_loss': train_stats['mean_distortion_loss'],
                'train_proposed_loss': train_stats['mean_proposed_loss'],
                'train_total_loss': train_stats['mean_total_loss'],
                'test_psnr': test_psnr,
                'test_msssim': test_msssim,
                'test_accuracy': test_accuracy,
                'train_runtime_seconds': train_runtime,
                'evaluation_runtime_seconds': evaluation_runtime,
                'epoch_runtime_seconds': epoch_runtime,
                'peak_allocated_mib': peak_allocated,
                'peak_reserved_mib': peak_reserved,
                'has_nan': train_stats['has_nan'],
                'has_inf': train_stats['has_inf'],
                'backward_success': train_stats['backward_success'],
                'optimizer_step_success': train_stats['optimizer_step_success'],
            }
            pilot_history.append(epoch_result)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'pilot_history': pilot_history,
                'configuration': vars(args),
            }
            save_intermediate = not (args.resume_pilot and args.eval_final_only)
            if save_intermediate or epoch + 1 == config.tot_epoch:
                torch.save(checkpoint, pilot_checkpoint)
            if epoch + 1 == config.tot_epoch:
                torch.save(checkpoint, './saved_model/pilot/epoch_{:03d}.pt'.format(epoch + 1))

            test_psnr_text = 'N/A' if test_psnr is None else '{:.9f}'.format(test_psnr)
            test_msssim_text = 'N/A' if test_msssim is None else '{:.9f}'.format(test_msssim)
            test_accuracy_text = 'N/A' if test_accuracy is None else '{:.6f}'.format(test_accuracy)
            print('[Pilot Epoch] epoch={} lr={} train_distortion={:.9f} train_proposed={:.9f} train_total={:.9f} test_psnr={} test_msssim={} test_accuracy={} train_runtime_seconds={:.3f} evaluation_runtime_seconds={:.3f} peak_allocated_mib={:.2f} peak_reserved_mib={:.2f}'.format(
                epoch + 1, cur_lr, train_stats['mean_distortion_loss'],
                train_stats['mean_proposed_loss'], train_stats['mean_total_loss'],
                test_psnr_text, test_msssim_text, test_accuracy_text,
                train_runtime, evaluation_runtime,
                peak_allocated, peak_reserved), flush=True)

        pilot_total_runtime = time.perf_counter() - pilot_total_start
        print('[Pilot Training Result]')
        print('Epoch | Train Loss | PSNR | MS-SSIM | Accuracy')
        for result in pilot_history:
            print('{} | {:.9f} | {} | {} | {}'.format(
                result['epoch'], result['train_total_loss'],
                'N/A' if result['test_psnr'] is None else '{:.9f}'.format(result['test_psnr']),
                'N/A' if result['test_msssim'] is None else '{:.9f}'.format(result['test_msssim']),
                'N/A' if result['test_accuracy'] is None else '{:.6f}'.format(result['test_accuracy'])))
        evaluated_results = [result for result in pilot_history if result['test_psnr'] is not None]
        print('best_psnr:', max(result['test_psnr'] for result in evaluated_results))
        print('best_msssim:', max(result['test_msssim'] for result in evaluated_results))
        print('best_accuracy:', max(result['test_accuracy'] for result in evaluated_results))
        resumed_results = [result for result in pilot_history if result['epoch'] > pilot_start_epoch]
        print('active_training_runtime_seconds: {:.3f}'.format(
            sum(result.get('train_runtime_seconds', 0.0) for result in resumed_results)))
        print('evaluation_runtime_seconds: {:.3f}'.format(
            sum(result.get('evaluation_runtime_seconds', 0.0) for result in resumed_results)))
        print('total_runtime_seconds: {:.3f}'.format(pilot_total_runtime))
        print('peak_allocated_mib:', max(result['peak_allocated_mib'] for result in resumed_results))
        print('peak_reserved_mib:', max(result['peak_reserved_mib'] for result in resumed_results))
        print('latest_checkpoint:', pilot_checkpoint)
        print('final_checkpoint: ./saved_model/pilot/epoch_{:03d}.pt'.format(config.tot_epoch))
    elif args.training:
        lambda_loss = args.lambda_loss
        
        for epoch in range(steps_epoch, config.tot_epoch):  
     
            if epoch < 200:
                cur_lr = 0.01
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 400:
                cur_lr = 0.005
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 550:
                cur_lr = 0.002
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 650:
                cur_lr = 0.001
                optimizer = optim.Adam(model_params, lr=cur_lr)
            elif epoch < 750:
                cur_lr = 0.0005
                optimizer = optim.Adam(model_params, lr=cur_lr)
            else:
                cur_lr = 0.0001
                optimizer = optim.Adam(model_params, lr=cur_lr)             


            lambda_loss = train_one_epoch(args, lambda_loss, H_fading_all)

            if (epoch + 1) % config.save_model_freq == 0:
                if args.channel_type == 'awgn':
                    save_model(net, save_path='./saved_model/awgn/STL10/SC_loss_snr{}_C{}.model'.format(snr, args.C))
                else:
                    save_model(net, save_path='./saved_model/rayleigh/STL10/SC_loss_snr{}_C{}.model'.format(snr, args.C))

                test(H_fading_all)

                np_train_PSNR_all = np.array(train_PSNR_all)
                np_train_Acc_all = np.array(train_Acc_all)
                np_train_SSIM_all = np.array(train_SSIM_all)
                np_val_PSNR_all = np.array(val_PSNR_all)
                np_val_Acc_all = np.array(val_Acc_all)
                np_val_SSIM_all = np.array(val_SSIM_all)
                np_test_PSNR_all = np.array(test_PSNR_all)
                np_test_Acc_all = np.array(test_Acc_all)
                np_test_SSIM_all = np.array(test_SSIM_all)

                if args.channel_type == 'awgn':
                    channel_str = 'results_SC_loss'
                else:
                    channel_str = 'results_SC_loss_Fading'

                file = ('./results_data/%s/train_PSNR_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_train_PSNR_all)

                file = ('./results_data/%s/train_Acc_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_train_Acc_all)

                file = ('./results_data/%s/train_SSIM_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_train_SSIM_all)

                file = ('./results_data/%s/val_PSNR_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_val_PSNR_all)

                file = ('./results_data/%s/val_Acc_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_val_Acc_all)

                file = ('./results_data/%s/val_SSIM_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_val_SSIM_all)

                file = ('./results_data/%s/test_PSNR_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_test_PSNR_all)

                file = ('./results_data/%s/test_Acc_C%d_SNR%d_lambda%.1f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_test_Acc_all)

                file = ('./results_data/%s/test_SSIM_C%.2f_SNR%d_lambda%.2f.npy' % (channel_str, args.C, snr_ini, args.lambda_loss))
                np.save(file, np_test_SSIM_all)

    else:
        test(H_fading_all)

