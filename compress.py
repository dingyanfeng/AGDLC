import numpy as np
import os
import json
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
from models_torch import *
import argparse
import arithmeticcoding_fast
import struct
import shutil
import time

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def loss_function(pred, target):
    loss = 1 / np.log(2) * F.nll_loss(pred, target)
    return loss


def compress(model, X_dict, Y, seq_1, bs, vocab_dict, timesteps, device, optimizer, scheduler, final_step=False):
    if not final_step:
        idx = max([(timesteps - 1) * int(str(num).split('.')[0]) + int(str(num).split('.')[1]) for num, _ in vocab_dict.items()])
        num_iters = FLAGS.data_len // bs
        ind = np.array(range(bs)) * num_iters

        f = [open(FLAGS.temp_file_prefix + '.' + str(i), 'wb') for i in range(bs)]
        bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
        enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

        prob = np.ones(4) / 4
        cumul = np.zeros(4 + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        # Encode first K symbols in each stream with uniform probabilities
        for i in range(bs):
            for j in range(min(idx, num_iters)):
                enc[i].write(cumul, seq_1[ind[i]+j])

        cumul = np.zeros((bs, 5), dtype=np.uint64)

        test_loss = 0
        batch_loss = 0
        train_loss = 0
        start_time = time.time()
        for j in (range(num_iters - idx)):
            bx = {num: Variable(torch.from_numpy(X_dict[num][ind, :])).to(device) for num, _ in vocab_dict.items()}
            by = Variable(torch.from_numpy(Y[ind])).to(device)

            with torch.no_grad():
                model.eval()
                pred = model(bx)
                loss = loss_function(pred, by)
                test_loss += loss.item()
                batch_loss += loss.item()
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

            for i in range(bs):
                enc[i].write(cumul[i, :], Y[ind[i]])

            # if (j + 1) % 100 == 0:
            #     print("{} secs".format(time.time() - start_time))
            #     print("Iter {} Loss {:.4f} Moving Loss {:.4f} Train Loss {:.4f}".format(j + 1, test_loss / (j + 1),
            #                                                                             batch_loss / 100,
            #                                                                             train_loss / 5), flush=True)
            #     batch_loss = 0
            #     train_loss = 0
            #     start_time = time.time()
            
            block_len = 20
            if (j + 1) % block_len == 0:
                indices = np.concatenate([ind - p for p in range(block_len)], axis=0)

                bx = {num: Variable(torch.from_numpy(X_dict[num][indices, :])).to(device) for num, _ in vocab_dict.items()}
                by = Variable(torch.from_numpy(Y[indices])).to(device)
                model.train()
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_function(pred, by)
                train_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

            ind += 1

        for i in range(bs):
            enc[i].finish()
            bitout[i].close()
            f[i].close()
    else:
        idx = max([(timesteps - 1) * int(str(num).split('.')[0]) + int(str(num).split('.')[1]) for num, _ in vocab_dict.items()])
        l = int(FLAGS.data_len / FLAGS.bs) * FLAGS.bs
        f = open(FLAGS.temp_file_prefix + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(4) / 4
        cumul = np.zeros(5, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(idx):
            enc.write(cumul, seq_1[l+j])
        for i in (range(len(Y))):
            # bx = Variable(torch.from_numpy(X[i:i + 1, :])).to(device)
            bx = {num: Variable(torch.from_numpy(X_dict[num][i:i + 1, :])).to(device) for num, _ in vocab_dict.items()}
            with torch.no_grad():
                model.eval()
                pred = model(bx)
                prob = torch.exp(pred).detach().cpu().numpy()
            cumul[1:] = np.cumsum(prob * 10000000 + 1)
            enc.write(cumul, Y[i])
        enc.finish()
        bitout.close()
        f.close()

    return


def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--skmer_list', type=str, default='3.4',
                        help='List of all avaliable s.k, connected with "+".')
    parser.add_argument('--file_name', type=str, default='Test',
                        help='The name of the input file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to use')
    parser.add_argument('--output', type=str, default='Output/Test',
                        help='Name of the output file')
    parser.add_argument('--module_type', type=str, default='LSTM',
                        help='module type')
    parser.add_argument('--timesteps', type=int, default='16',
                        help='Number of timesteps')
    parser.add_argument('--bs', type=int, default='320',
                        help='Batch Size')
    return parser


def var_int_encode(byte_str_len, f):
    while True:
        this_byte = byte_str_len & 127
        byte_str_len >>= 7
        if byte_str_len == 0:
            f.write(struct.pack('B', this_byte))
            break
        f.write(struct.pack('B', this_byte | 128))
        byte_str_len -= 1


def get_elements(arr, idx_range, k, step, timesteps):
    result = np.empty((len(idx_range), timesteps), dtype=arr.dtype)
    for idx, i in enumerate(idx_range):
        end_index = max(0, k + i - 1 - ((timesteps - 1) * step))  # 计算起始索引
        start_index = k + i  # 结束索引为第i个元素的下一个索引
        series_data = arr[start_index:end_index:-step]
        
        if k + i - ((timesteps - 1) * step) == 0:
            series_data = np.append(series_data, arr[0])
        if len(series_data) != timesteps:
            raise ValueError("Length of get_elements result is not timesteps.")
        result[idx] = series_data[::-1]  # 使用切片操作获取元素
    return result


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    batch_size = FLAGS.bs
    timesteps = FLAGS.timesteps
    use_cuda = True
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    skmer_list= [float(num) for num in FLAGS.skmer_list.split('+')]
    FLAGS.temp_dir = 'temp'
    if os.path.exists(FLAGS.temp_dir):
        os.system("rm -r {}".format(FLAGS.temp_dir))
    FLAGS.temp_file_prefix = FLAGS.temp_dir + "/compressed"
    if not os.path.exists(FLAGS.temp_dir):
        os.makedirs(FLAGS.temp_dir)

    # 准备参数
    vocab_dict = {}
    emb_dict = {}
    hidden_dict = {}
    # ————————
    for number in skmer_list:
        _, k = str(number).split('.')
        vocab_dict[number] = pow(4, int(k))
        emb_dict[number] = 16
        hidden_dict[number] = 128

    FLAGS.param = "Params-Seq/params_" + os.path.splitext(FLAGS.file_name)[0] + "_" + str(1)
    with open(FLAGS.param, 'r') as f:
        params = json.load(f)
    params['bs'] = batch_size
    params['timesteps'] = timesteps
    with open(FLAGS.output + '.params', 'w') as f:
        json.dump(params, f, indent=4)

    # idx = n * s + k - 1   (n=0,1,2...; idx是从0开始得到的结果)
    pred_sequence = np.load("Params-Seq/" + FLAGS.file_name + "_" + str(1) + ".npy")
    FLAGS.data_len = len(pred_sequence)
    idx = max([(timesteps - 1) * int(str(num).split('.')[0]) + int(str(num).split('.')[1]) for num in skmer_list])
    X_dict = {}
    Y = pred_sequence[idx:idx+FLAGS.data_len].copy()
    for num in skmer_list:
        s, k = str(num).split('.')
        sequence = np.load("Params-Seq/" + FLAGS.file_name + "_" + k + ".npy")
        sequence = sequence
        sequence = sequence.reshape(-1)
        series = sequence.copy()
        X_dict[num] = get_elements(series, range(0, FLAGS.data_len - idx), idx-int(k), int(s), timesteps)

    module_type = FLAGS.module_type
    cskmerdic = {'skmer_list': skmer_list, 'vocab_sizes': vocab_dict, 'emb_sizes': emb_dict,
                 'hidden_sizes': hidden_dict, 'seq_length': timesteps, 'module_type':module_type}
    # print(cskmerdic)
    # Define Model
    cskmerm = MultiLSTMModel(**cskmerdic).to(device)
    # print(cskmerm)

    optimizer = optim.Adam(cskmerm.parameters(), lr=5e-4, betas=(0.0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=1e-2, patience=1000,
                                                     cooldown=10000, min_lr=1e-4)

    l = int(len(pred_sequence) / batch_size) * batch_size

    compress(cskmerm, X_dict, Y, pred_sequence, batch_size, vocab_dict, timesteps, device, optimizer, scheduler)
    if l < len(pred_sequence) - idx:
        compress(cskmerm, {num: X_dict[num][l:] for num,_ in X_dict.items()}, Y[l:], pred_sequence, 1, vocab_dict, timesteps, device, optimizer, scheduler, final_step=True)
    else:
        f = open(FLAGS.temp_file_prefix + '.last', 'wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(4) / 4

        cumul = np.zeros(5, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)
        for j in range(l, len(pred_sequence)):
            enc.write(cumul, pred_sequence[j])
        enc.finish()
        bitout.close()
        f.close()
    
    # combine files into one file
    f = open(FLAGS.output + '.combined', 'wb')
    for i in range(batch_size):
        f_in = open(FLAGS.temp_file_prefix + '.' + str(i), 'rb')
        byte_str = f_in.read()
        byte_str_len = len(byte_str)
        var_int_encode(byte_str_len, f)
        f.write(byte_str)
        f_in.close()
    f_in = open(FLAGS.temp_file_prefix + '.last', 'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
    f.close()
    shutil.rmtree('temp')
    print("Done")

    print('Peak GPU memory usage:{} KBs'.format(torch.cuda.max_memory_allocated() // 1024))


if __name__ == "__main__":
    parser = get_argument_parser()
    FLAGS = parser.parse_args()
    main()