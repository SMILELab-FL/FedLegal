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
  config_file="config_files/${task,,}/eval_f1_macro/cen.yaml"
elif [[ $task == "LAM" ]]; then
  config_file="config_files/${task,,}/eval_f1_micro/cen.yaml"
else
  config_file="config_files/${task,,}/cen.yaml"
fi
echo "config file in ${config_file}"


bash cen_run.sh ${RUN_DIR} ${task} centralized ${config_file} 18000 0