from unicodedata import category
import argparse
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import warnings
import pdb


warnings.filterwarnings("ignore")


def downsampling(input, out_size):
    downsampled_data = torch.nn.functional.interpolate(input,size=(out_size, out_size),mode='bilinear')
    return downsampled_data


def data_tf(x):
    x = x.resize((96, 96), 2) 
    x = np.array(x, dtype='float32') / 255
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    config = parser.parse_args()

    CBsize = 64  # codebook size, 8 16 32 64

    torch.manual_seed(1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.smoke_test and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    os.makedirs('./results_data', exist_ok=True)
    print('device:', device)   
    batchsize = CBsize
    epoch_len = 2 if config.smoke_test else 500
    codebook = None
    distance_measure = nn.MSELoss()
    counter = np.ones(CBsize)
    iteration_interval = 1 if config.smoke_test else 50
    images_per_batch = 1 if config.smoke_test else batchsize
    codebook_initial_shape = None
    assignment_success = False
    update_success = False
    start_time = time.perf_counter()

    train_set = datasets.STL10("./media/Dataset/CIFAR10/", transform=data_tf, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True)

    if config.smoke_test:
        print('dataset load success:', len(train_set) > 0)
        print('dataset size:', len(train_set))

    print('Codebook Construction Start!')
    
    
    for e in range(epoch_len):
        print('epoch:', e)
        iteration = 0
        for im, label in train_loader:
            iteration += 1
            im = Variable(im)  # batchsize
            im = downsampling(im, 256)
            im = im.to(device)
            # initialize the codebook
            if e == 0:
                codebook = im.clone()
                codebook_initial_shape = tuple(codebook.shape)
                print('codebook initialization is done ...')
                break

            distance_min = 10 ** 8
            category = 0  # the corresponding codeword index

            for i in range(images_per_batch):
                for j in range(CBsize):
                    distance = distance_measure(im[i], codebook[j])
                    if distance < distance_min:
                        distance_min = distance
                        category = j
                assignment_success = True
                counter[category] += 1
                codebook[category] = torch.add(codebook[category] * (counter[category] - 1) / counter[category], im[i] / counter[category])
                update_success = True

            if iteration == iteration_interval: 
                break
        
        print('counters:', counter)

        if (config.smoke_test and e == epoch_len - 1) or (not config.smoke_test and e % 10 == 0):
            print('save the codebook ...')
            codebook0 = codebook.clone()
            codebook0 = codebook0.view(CBsize, int(3 * 256 * 256))
            np_codebook = codebook0.detach().cpu().numpy()

            if config.smoke_test:
                file = './results_data/codebook_smoke.npy'
            else:
                file = ('./results_data/codebook_size%d.npy' % (CBsize))
            np.save(file, np_codebook)

    if config.smoke_test:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            max_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            max_vram_mib = 0.0
        elapsed = time.perf_counter() - start_time
        print('Smoke test results:')
        print('  classifier load success: N/A')
        print('  codebook initial shape:', codebook_initial_shape)
        print('  final codebook shape:', tuple(np_codebook.shape))
        print('  dtype:', np_codebook.dtype)
        print('  has NaN:', bool(np.isnan(np_codebook).any()))
        print('  has Inf:', bool(np.isinf(np_codebook).any()))
        print('  codeword assignment success:', assignment_success)
        print('  update success:', update_success)
        print('  saved:', os.path.isfile(file), '(' + file + ')')
        print('  execution time: {:.3f} seconds'.format(elapsed))
        print('  GPU max VRAM: {:.2f} MiB'.format(max_vram_mib))




