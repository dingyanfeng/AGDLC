#!/bin/bash
data=$1
GPU=$2
echo ${data}
base_name=$(basename ${data})
base_name_without_extension=$(echo "$base_name" | cut -d. -f1)
# 创建保存结果的目录
data_path=/home/dyf/Dynamic-skmer/Result_ns/sklist/${base_name_without_extension}
if [ ! -d "${data_path}" ]; then
  mkdir -p "${data_path}"
  echo "Created directory: ${data_path}"
else
  echo "Directory already exists: ${data_path}"
fi
sum_result="${data_path}/sum_result.csv"
echo "ALG, DS(B), CS(B), CR(bits/base), CT(S), CPM(KB), DT(S), DPM(KB), GPU-Mem(KB), Throughtput(KB/S)" >> ${sum_result}
# 时间转换函数
function timer_reans() {
  if [[ $1 == *"."* ]]; then
    local min=$(echo "$1" | cut -d ':' -f 1)
    local sec=$(echo "$1" | cut -d ':' -f 2 | cut -d '.' -f 1)
    local ms=$(echo "$1" | cut -d '.' -f 2)
    local result=$(echo $min $sec $ms | awk '{printf ("%.3f\n", 60*$1+$2+$3/1000+1)}')
    echo $result
  else
    local hour=$(echo "$1" | cut -d ':' -f 1)
    local min=$(echo "$1" | cut -d ':' -f 2)
    local sec=$(echo "$1" | cut -d ':' -f 3)
    local result=$(echo $hour $min $sec | awk '{printf ("%.3f\n",3600*$1+60*$2+$3+1.001)}')
    echo $result
  fi
}

# 针对单个算法创建的LOG文件
function LogFile() {
  alg_path=${data_path}/$1 # $1: 算法名称
  if [ ! -d "${alg_path}" ]; then
    mkdir -p "${alg_path}"
  fi
}

function DSKmer() {
  T_SKLIST=$1
  T_MODULE=$2
  echo "Running DSKmer algorithm"
  echo "1 Create log file..."
  LogFile ${T_MODULE}
  echo "2 Execute compression task..."
  log=${data_path}/$2/comp-$1.log
  (/bin/time -v -p sh /home/dyf/Dynamic-skmer/compress.sh ${data} ${T_SKLIST} ${T_MODULE}) >${log} 2>&1

  if [ $? -ne 0 ]; then
    echo "compression task ERROR!"
    echo "/bin/time -v -p sh /home/dyf/Dynamic-skmer/compress.sh ${data} ${T_SKLIST} ${T_MODULE}"
    exit 1
  fi
  echo "3 Statistical compression information..."
  CompressedFileSize_File=$(ls -lah --block-size=1 /home/dyf/Dynamic-skmer/CompRes/${base_name_without_extension}.combined | awk '/^[-d]/ {print $5}')
  CompressedFileSize_Params=$(ls -lah --block-size=1 /home/dyf/Dynamic-skmer/CompRes/${base_name_without_extension}.params | awk '/^[-d]/ {print $5}')
  CompressedFileSize=$(echo "scale=3; ${CompressedFileSize_File}+${CompressedFileSize_Params}" | bc)
  CompressionTime=$(cat ${log} | grep -o 'Elapsed (wall clock) time (h:mm:ss or m:ss):.*' | awk '{print $8}')
  GPUMem=$(cat ${log} | grep -o 'Peak GPU memory usage:.*' | awk '{print $5}')
  CompressionMemory=$(cat ${log} | grep -o 'Maximum resident set size.*' | grep -o '[0-9]*')
  SourceFileSize=$(ls -lah --block-size=1 ${data} | awk '/^[-d]/ {print $5}') #以字节为单位显示原始文件大小
  CompressionRatio=$(echo "scale=3; 8*${CompressedFileSize}/${SourceFileSize}" | bc)
  #CompressionRatio=$(echo $CompressedFileSize $SourceFileSize | awk '{printf ("%.3f\n", 8*$1/$2)}')
  echo "CompressedFileSize : ${CompressedFileSize} B"
  echo "CompressedFileSize_File : ${CompressedFileSize_File} B"
  echo "CompressedFileSize_Params : ${CompressedFileSize_Params} B"
  echo "CompressionTime : ${CompressionTime} h:mm:ss or m:ss"
  echo "CompressionTime : $(timer_reans $CompressionTime) S"
  echo "CompressionMemory : ${CompressionMemory} KB"
  echo "GPUMemory : ${GPUMem} KB"
  echo "SourceFileSize : ${SourceFileSize} B"
  echo "CompressionRatio : ${CompressionRatio} bits/base"
  echo "4 Execute decompression task..."
  log=${data_path}/$2/decomp-$1.log
  (/bin/time -v -p sh /home/dyf/Dynamic-skmer/decompress.sh ${data} ${T_SKLIST} ${T_MODULE}) >${log} 2>&1
  if [ $? -ne 0 ]; then
    echo "decompression task ERROR!"
    echo "/bin/time -v -p sh /home/dyf/Dynamic-skmer/decompress.sh ${data} ${T_SKLIST} ${T_MODULE}"
    exit 1
  fi
  echo "5 Statistical decompression information..."
  DeCompressionTime=$(cat ${log} | grep -o 'Elapsed (wall clock) time (h:mm:ss or m:ss):.*' | awk '{print $8}')
  DeCompressionMemory=$(cat ${log} | grep -o 'Maximum resident set size.*' | grep -o '[0-9]*')
  echo "DeCompressionTime : ${DeCompressionTime} h:mm:ss or m:ss"
  echo "DeCompressionTime : $(timer_reans $DeCompressionTime) S"
  echo "DeCompressionMemory : ${DeCompressionMemory} KB"

  TRP=$(echo "scale=3; ${SourceFileSize}/($(timer_reans $DeCompressionTime)+$(timer_reans $CompressionTime))/1024" | bc)
  echo "Throughtput : ${TRP} KB/S"
  echo "6 Output file into ${sum_result}..."
  # "ALG, DS(B), CS(B), CR(bits/base), CT(S), DT(S), CPM(KB), DPM(KB)"
  echo "${T_SKLIST}_${T_MODULE}, ${SourceFileSize}, ${CompressedFileSize}, ${CompressionRatio}, $(timer_reans $CompressionTime), ${CompressionMemory}, $(timer_reans $DeCompressionTime), ${DeCompressionMemory} ${GPUMem} ${TRP}" >>${sum_result}
  echo "7 Check the integrity of the decompressed file..."
  python /home/dyf/Dynamic-skmer/Script/tool.py cmp /home/dyf/Dynamic-skmer/DecompRes/${base_name_without_extension} ${data}
  if [ $? -ne 0 ]; then
    echo "Decompressed file verification failed!"
    #exit 1
    echo "${T_SKLIST}err! try again"  >>${sum_result}
  fi
  echo "8 Delete file..."
  rm -rf CompRes/${base_name_without_extension}*
  rm -rf DecompRes/${base_name_without_extension}
  rm -rf Params-Seq/${base_name_without_extension}*
  rm -rf Params-Seq/params_${base_name_without_extension}*
#   cd ${pwd_path}
}

# echo "************************************************"
# DSKmer ${data} ${SKLIST} ${MODULE_TYPE}
# echo "************************************************"
for (( i = 1; i <= 4; i++ )); do
    echo "DSKmer ${i}.${i} LSTM"
    DSKmer ${i}.${i} LSTM
done
cat ${sum_result}
exit 0