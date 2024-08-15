import argparse
import sys
import os
import numpy as np
import re

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Tool Commands')
    # fasta2txt
    fasta2txt_parser = subparsers.add_parser(name='fasta2txt', help='extracting DNA from fasta file')
    fasta2txt_parser.add_argument('file1', type=str, help='data_path_1')
    fasta2txt_parser.add_argument('file2', type=str, help='data_path_2')
    fasta2txt_parser.set_defaults(func=data)
    # processing data streams
    data_parser = subparsers.add_parser(name='data', help='dealing data streams as pure single ACGT string')
    data_parser.add_argument('file1', type=str, help='data_path_1')
    data_parser.add_argument('file2', type=str, help='data_path_2')
    data_parser.set_defaults(func=data)
    # compare
    cmp_parser = subparsers.add_parser(name='cmp', help='Compare two files for consistency.')
    cmp_parser.add_argument('file1', type=str, help='File 1.')
    cmp_parser.add_argument('file2', type=str, help='File 2.')
    cmp_parser.set_defaults(func=cmp)
    # compression ratio
    cr_parser = subparsers.add_parser(name='cr', help='Compute compression ratio.')
    cr_parser.add_argument('file1', type=str, help='Source file.')
    cr_parser.add_argument('file2', type=str, help='Compressed file.')
    cr_parser.set_defaults(func=cr)
    # compare & cr
    p_parser = subparsers.add_parser(name='perform', help='Compute compression ratio.')
    p_parser.add_argument('file1', type=str, help='Source file.')
    p_parser.add_argument('file2', type=str, help='Compressed file.')
    p_parser.add_argument('file3', type=str, help='Decompressed file.')
    p_parser.set_defaults(func=perform)
    args = parser.parse_args(argv)
    args.func(args)
    return args

def compare(file1, file2):
    with open(file1, 'rb') as f:  # 一次一个byte = 8bit
        series1 = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()
    with open(file2, 'rb') as f:  # 一次一个byte = 8bit
        series2 = np.frombuffer(f.read(), dtype=np.uint8)
    f.close()
    if len(series1) == len(series2) and (series1==series2).all():
        return True
    else:
        return False

def compute_cr(file1, file2):
    f1_size, f2_size = os.stat(file1).st_size, os.stat(file2).st_size
    return round(f2_size/f1_size*8, 5)

def cmp(args):      # 比较decompressed file与source file是否一致
    file1, file2 = args.file1, args.file2
    if compare(file1, file2):
        print('The file {} is the same as {}'.format(file1, file2))
        exit(0)
    else:
        print('The file {} is different from {}'.format(file1, file2))
        exit(1)

def fasta2txt(args):      # 提取FastA的所有序列
    file1, file2 = args.file1, args.file2
    sequences = []
    #print("hahhah")
    with open(file1, 'r') as file:
        sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    with open(file2, 'w') as file:
        for i, sequence in enumerate(sequences):
            #file.write(f'>Sequence {i + 1}\n')
            file.write(sequence + '\n')
    print("Total Number: ", len(sequences))


def cr(args):       # 计算compressed file与source file之间的压缩率
    file1, file2 = args.file1, args.file2
    print('The compression ratio between <ori:{}> and <comp:{}> is {}.'.format(file1, file2, compute_cr(file1, file2)))

def perform(args):  # 比较source与decomp的一致性，计算source与comp的压缩率
    file1, file2, file3 = args.file1, args.file2, args.file3
    if compare(file1, file3):
        print('The file {} is the same as {}'.format(file1, file3))
    else:
        print('The file {} is different from {}'.format(file1, file3))
    print('The compression ratio between <ori:{}> and <comp:{}> is {}.'.format(file1, file2, compute_cr(file1, file2)))

def data(args):      # 将数据处理为仅有ACGT的字符串
    file1, file2 = args.file1, args.file2
    with open(file1, 'r') as source_file:
        source = source_file.read()
    source = source.upper() # 小写转大写
    source = re.sub(r'[^ACGT]', '', source) # 剔除非ACGT字符串
    with open(file2, 'w') as file:
        file.write(source)

if __name__ == '__main__':
    parseArgs(sys.argv[1:])