# AGDLC

### About The AGDLC

**AGDLC** is an experimental learning-based genomics data lossless compressor, which utilize xLSTM-based context modeling and multiple (*s*,*k*)-mer encoding. The code can run on GPU platform.

### Requirements

0. GPU

1. Python 3.11.9

2. xlstm module ( See [NX-AI/xlstm](https://github.com/NX-AI/xlstm))

### Copy Our Project

Firstly, clone our tools from GitHub:

```
git clone https://github.com/dingyanfeng/AGDLC.git
```

Secondly, turn to AGDLC directory:

```
cd AGDLC
```

### Usage

#### 1.compress with AGDLC

```bash
bash compress.sh [FILE TO COMPRESS] [SKMER LIST] xLSTM [GPU NUMBER] [TIMESTEPS]
# [SKMER LIST]: All (s,k)-mer to be merged, use the form 's.k' and connected with '+'.
# [GPU NUMBER]: Choose which GPU to run the code.
# [TIMESTEPS] : The feature length extracted by each (s,k)-mer.
```

#### 2.decompress with AGDLC

```bash
bash decompress.sh [COMPRESSED FILE] [SKMER LIST] xLSTM [GPU NUMBER]
# [COMPRESSED FILE]: The output file of compressor.
```

### Examples

#### 1.compress with AGDLC

```bash
bash compress.sh DataSets/USE/AeCa 2.3+3.3 xLSTM 1 32
```

#### 2.decompress with AGDLC

```bash
bash decompress.sh CompRes/AeCa 2.3+3.3 xLSTM 1
```

### Our Experimental Configuration

The experiment is running on Linux system  with:

2*Intel Xeon Silver 4310 CPUs

4*NVIDIA GeForce RTX 4090 GPUs (24GB CUDA memory)

128 GB DDR5 memory

### Credits

The arithmetic coding is performed using the code available at [Reference-arithmetic-coding](https://github.com/nayuki/Reference-arithmetic-coding). The code is a part of Project Nayuki.

The xLSTM model is performed using the code available at [NX-AI/xlstm](https://github.com/NX-AI/xlstm). 
