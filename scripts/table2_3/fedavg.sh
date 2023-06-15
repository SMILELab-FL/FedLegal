RUN_DIR=$1
task=${2^^}

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"

TASK_LIST=('LCP' 'LJP' 'LER' 'LRE' 'LAM')
if [[ ! " ${TASK_LIST[@]} " =~ " ${task} " ]]; then
  echo "Error task input ${Task}. Task value should be one of ${TASK_LIST[@]}"
  exit 1
fi


if [[ $task == "LRE" ]]; then
  config_file="config_files/${task,,}/eval_f1_macro/fed_silo.yaml"
elif [[ $task == "LAM" ]]; then
  config_file="config_files/${task,,}/eval_f1_micro/fed_silo.yaml"
else
  config_file="config_files/${task,,}/fed_silo.yaml"
fi
echo "config file in ${config_file}"


bash fed_run.sh ${RUN_DIR} ${task} fedavg ${config_file} 18000 0,0,1
