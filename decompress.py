import numpy as np
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from models_torch import *
from utils import *
import tempfile
import argparse
import arithmeticcoding_fast
import struct
import time
import shutil

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def loss_function(pred, target):
    loss = 1/np.log(2) * F.nll_loss(pred, target)
    return loss


def decompress(model, len_series, bs, vocab_dict, timesteps, device, optimizer, scheduler, final_step=False):
    idx = max([(timesteps - 1) * int(str(num).split('.')[0]) + int(str(num).split('.')[1]) for num, _ in vocab_dict.items()])
    if not final_step:
        num_iters = len_series // bs
        series_2d = np.zeros((bs,num_iters), dtype = np.uint8).astype('int')

        f = [open(FLAGS.temp_file_prefix+'.'+str(i),'rb') for i in range(bs)]
        bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
        dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

        prob = np.ones(4)/4
        cumul = np.zeros(5, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        # Decode first K symbols in each stream with uniform probabilities
        for i in range(bs):
            for j in range(min(idx, num_iters)):
                series_2d[i,j] = dec[i].read(cumul, 4)

        cumul = np.zeros((bs, 5), dtype = np.uint64)

        block_len = 20
        test_loss = 0
        batch_loss = 0
        start_time = time.time()
        for j in (range(num_iters - idx)):
            bx_data = series_2d[:,j:j+idx]
            bx = {}
            for num, _ in vocab_dict.items():
                s, k = map(int, str(num).split('.'))
                new_array = np.flip(np.array([np.sum(bx_data[:, -(iter*s + 1):-(iter*s + k + 1):-1] * np.power(4, np.arange(k)), axis=1) for iter in range(timesteps)]).T, axis=1)
                copied_new_array = new_array.copy()
                bx[num] = Variable(torch.from_numpy(copied_new_array)).to(device)
            with torch.no_grad():
                model.eval()
                pred = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

            # Decode with Arithmetic Encoder
            for i in range(bs):
                series_2d[i,j+idx] = dec[i].read(cumul[i,:], 4)

            by = Variable(torch.from_numpy(series_2d[:, j+idx])).to(device)
            loss = loss_function(pred, by)
            test_loss += loss.item()
            batch_loss += loss.item()

            # if (j+1) % 100 == 0:
            #     print("Iter {} Loss {:.4f} Moving Loss {:.4f}".format(j+1, test_loss/(j+1), batch_loss/100), flush=True)
            #     print("{} secs".format(time.time() - start_time))
            #     batch_loss = 0
            #     start_time = time.time()

            # Update Parameters of Combined Model
            if (j+1) % block_len == 0:
                model.train()
                optimizer.zero_grad()
                bx_data_x = np.concatenate([series_2d[:, j + np.arange(idx) - p] for p in range(block_len)], axis=0)
                bx = {}
                for num, _ in vocab_dict.items():
                    s, k = map(int, str(num).split('.'))
                    data_x = np.flip(np.array([np.sum(bx_data_x[:, -(i*s + 1):-(i*s + k + 1):-1] * np.power(4, np.arange(k)), axis=1) for i in range(timesteps)]).T, axis=1)
                    copied_data_x = data_x.copy()
                    bx[num] = Variable(torch.from_numpy(copied_data_x)).to(device)
                data_y = np.concatenate([series_2d[:, j + idx - p] for p in range(block_len)], axis=0)
                by = Variable(torch.from_numpy(data_y)).to(device)
                pred = model(bx)
                loss = loss_function(pred, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
        
        # close files
        for i in range(bs):
            bitin[i].close()
            f[i].close()
        return series_2d.reshape(-1)
    
    else:
        series = np.zeros(len_series, dtype = np.uint8).astype('int')
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
        prob = np.ones(4)/4
        cumul = np.zeros(5, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        

        for j in range(min(idx,len_series)):
            series[j] = dec.read(cumul, 4)
        for i in range(len_series-idx):
            bx_data = series[i:i+idx].reshape(1,-1)
            bx = {}
            for num, _ in vocab_dict.items():
                s, k = map(int, str(num).split('.'))
                data_x = np.flip(np.array([np.sum(bx_data[:, -(i*s + 1):-(i*s + k + 1):-1] * np.power(4, np.arange(k)), axis=1) for i in range(timesteps)]).T, axis=1)
                copied_data_x = data_x.copy()
                bx[num] = Variable(torch.from_numpy(copied_data_x)).to(device)
            with torch.no_grad():
                model.eval()
                pred = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob*10000000 + 1)
            series[i+idx] = dec.read(cumul, 4)
        bitin.close()
        f.close()
        return series


def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--skmer_list', type=str, default='3.4',
                        help='List of all avaliable s.k, connected with "+".')
    parser.add_argument('--file_name', type=str, default='Output/Test',
                        help='The name of the input file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    parser.add_argument('--output', type=str, default='Test',
                        help='Name of the output file')
    parser.add_argument('--module_type', type=str, default='LSTM',
                        help='Module Type')
    parser.add_argument('--emb_size', type=int, default='16',
                        help='Embedding Size')
    return parser


def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device {}".format(device))

    FLAGS.temp_dir = 'temp'
    if os.path.exists(FLAGS.temp_dir):
        shutil.rmtree('temp')
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    f = open(FLAGS.file_name+'.params','r')
    params = json.loads(f.read())
    f.close()

    batch_size = params['bs']
    timesteps = params['timesteps']
    len_series = params['Length-Data']
    id2char_dict = params['id2char_dict']

    # Break into multiple streams
    f = open(FLAGS.file_name+'.combined','rb')
    for i in range(batch_size):
        f_out = open(FLAGS.temp_file_prefix+'.'+str(i),'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
    f_out = open(FLAGS.temp_file_prefix+'.last','wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
    f.close()

    series = np.zeros(len_series,dtype=np.uint8)

    vocab_dict = {}
    emb_size = FLAGS.emb_size
    hidden_size = 128
    skmer_list= [float(num) for num in FLAGS.skmer_list.split('+')]
    for number in skmer_list:
        _, k = str(number).split('.')
        vocab_dict[number] = pow(4, int(k))

    module_type = FLAGS.module_type
    cskmerdic = {'skmer_list': skmer_list, 'vocab_sizes': vocab_dict, 'emb_sizes': emb_size,
                 'hidden_sizes': hidden_size, 'seq_length': timesteps, 'module_type':module_type}
    cskmerm = MultiLSTMModel(**cskmerdic).to(device)

    optimizer = optim.Adam(cskmerm.parameters(), lr=5e-4, betas=(0.0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=1e-2, patience=1000,
                                                     cooldown=10000, min_lr=1e-4)
    
    idx = max([(timesteps - 1) * int(str(num).split('.')[0]) + int(str(num).split('.')[1]) for num, _ in vocab_dict.items()])
    l = int(len(series)/batch_size)*batch_size
    series[:l] = decompress(cskmerm, l, batch_size, vocab_dict, timesteps, device, optimizer, scheduler)
    if l < len_series - idx:
        series[l:] = decompress(cskmerm, len_series-l, 1, vocab_dict, timesteps, device, optimizer, scheduler, final_step=True)
    else:
        f = open(FLAGS.temp_file_prefix+'.last','rb')
        bitin = arithmeticcoding_fast.BitInputStream(f)
        dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin) 
        prob = np.ones(4)/4
        
        cumul = np.zeros(5, dtype = np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)        
        for j in range(l, len_series):
            series[j] = dec.read(cumul, 4)
        
        bitin.close() 
        f.close()
    print(series, len(series))
    # Write to output
    f = open(FLAGS.output, 'w')
    for s in series:
        if s not in [0,1,2,3]:
            print(s)
        else:
            f.write(id2char_dict[str(s)])
    # f.write(bytearray([id2char_dict[str(s)] for s in series]))
    f.close()

    shutil.rmtree('temp')
    print("Done")

    print('Peak GPU memory usage: {} KBs'.format(torch.cuda.max_memory_allocated()//1024))


if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()