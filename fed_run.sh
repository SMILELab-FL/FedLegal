#!/bin/bash
# shellcheck disable=SC2068
# 读取参数
idx=0
for i in $@
do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

# 分离参数
run_dirs=${args[0]}
task_name=${args[1]}
fl_algorithm=${args[2]}
config_path=${args[3]}  # relative in this project
port=${args[4]}

idx=0
for(( i=5;i<${#args[@]};i++ ))
do
  device[${idx}]=${args[i]}
  let "idx=${idx}+1"
done

world_size=${#device[@]}
echo "world_size is ${world_size}"
echo "run_dirs: ${run_dirs}"

max_seq=512
# dataset_name = "legal"
# metric_name = "legal"
model_name='roberta-wwm-ext'
data_file='legal/silo'
if [ ${task_name} == "LCP" ];
then
  model_output_mode="seq_classification"
elif [ ${task_name} == "LJP" ];
then
  model_output_mode='seq_regression'
elif [ ${task_name} == "LAM" ];
then
  model_output_mode='multi_seq_classification'
elif [ ${task_name} == "LER" ];
then
  model_output_mode='token_classification_crf'
elif [ ${task_name} == "LRE" ];
then
  model_output_mode='seq_classification'
elif [ ${task_name} == "LDG" ];
then
  model_output_mode='seq_generation'
  model_name='gpt2-chinese-cluecorpussmall'
  max_seq=1024
else
  echo "Don't support ${task_name}"
  exit 1
fi
echo "${task_name}'s max_seq is ${max_seq}"


#sleep 7h

CUDA_VISIBLE_DEVICES=${device[0]} python main.py \
--model_name_or_path ${run_dirs}/pretrain/nlp/${model_name}/ \
--output_dir ${run_dirs}/output/${data_file} \
--rank 0 \
--task_name ${task_name} \
--fl_algorithm ${fl_algorithm} \
--raw_dataset_path ${run_dirs}/datasets/${data_file} \
--partition_dataset_path ${run_dirs}/datasets/${data_file} \
--max_seq_length ${max_seq} \
--world_size ${world_size} \
--port ${port} \
--config_path ${config_path} \
--test_rounds True &

#sleep 2s

for(( i=1;i<${world_size};i++))
do
{
    echo "client ${i} started"
    CUDA_VISIBLE_DEVICES=${device[i]} python main.py \
    --model_name_or_path ${run_dirs}/pretrain/nlp/${model_name}/ \
    --output_dir ${run_dirs}/output/${data_file} \
    --rank ${i} \
    --task_name ${task_name} \
    --fl_algorithm ${fl_algorithm} \
    --raw_dataset_path ${run_dirs}/datasets/${data_file} \
    --partition_dataset_path ${run_dirs}/datasets/${data_file} \
    --max_seq_length ${max_seq} \
    --world_size ${world_size} \
    --port ${port} \
    --config_path ${config_path} \
    --test_rounds True &
    
#    sleep 2s
}
done

wait
