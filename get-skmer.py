import sys
import numpy as np
import json
import argparse
import os
import re


# 得到k-mer编码词典
def get_dict(alphabet, k):
    combinations = []

    def generate_helper(current_combination, remaining_length):
        if remaining_length == 0:
            combinations.append(current_combination)
            return
        for letter in alphabet:
            generate_helper(current_combination + letter, remaining_length - 1)

    generate_helper("", k)
    char2id_dict = {c: i for (i, c) in enumerate(combinations)}
    id2char_dict = {i: c for (i, c) in enumerate(combinations)}
    return char2id_dict, id2char_dict


def get_argument_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('--file_name', type=str, default='DataSets/Test',
                        help='The name of the input file')
    parser.add_argument('--skmer_list', type=str, default='3.4',
                        help='List of all avaliable s.k, connected with "+".')
    
    return parser

parser = get_argument_parser()
FLAGS = parser.parse_args()

input_file = FLAGS.file_name

skmer_list= [float(num) for num in FLAGS.skmer_list.split('+')]
valid_sk = {'1', '2', '3', '4'}

base_name = os.path.basename(input_file)

param_dict = {}
output_dict = {}
k_list = []
for number in skmer_list:
    s, k = str(number).split('.')
    if s not in valid_sk or k not in valid_sk:
        raise ValueError(f"sk value {number} is invalid. Program exiting.")
    if int(k) not in k_list:
        k_list.append(int(k))
        param_dict[int(k)] = "Params-Seq/params_" + os.path.splitext(base_name)[0] + "_" + k
        output_dict[int(k)] = "Params-Seq/" + os.path.splitext(base_name)[0] + "_" + k
if 1 not in k_list:
    k_list.append(1)
    param_dict[1] = "Params-Seq/params_" + os.path.splitext(base_name)[0] + "_" + str(1)
    output_dict[1] = "Params-Seq/" + os.path.splitext(base_name)[0] + "_" + str(1)

print("input_file : ", input_file)
print("sk-mer list : ", skmer_list)
print("param_dict : ", param_dict)
print("output_dict : ", output_dict)

with open(input_file) as fp:
    data = fp.read()
data = re.sub(r'[^ACGT]', '', data)
print("Seq Length {}".format(len(data)))
print(k_list)

def encode_sequence(data, char2id_dict, k, w):
    res = []
    i = 0
    while i < len(data) - k + w:
        sub_ = str(data[i:i + k])
        if len(sub_) == k :
            res.append(char2id_dict[sub_])
        else:
            params['Write-Chars'] = sub_[k-w:]
        i = i + w
    return res
for k in k_list:
    char2id_dict, id2char_dict = get_dict(['A','C','G','T'], k)
    char2id_dict = {key: value for key, value in char2id_dict.items()}
    id2char_dict = {key: value for key, value in id2char_dict.items()}
    params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict}
    
    temp_data = encode_sequence(data, char2id_dict, k, 1)
    params['Length-Data'] = len(data)
    with open(param_dict[int(k)], 'w') as f:
        json.dump(params, f, indent=4)
    integer_encoded = np.array(temp_data)
    np.save(output_dict[int(k)], integer_encoded)