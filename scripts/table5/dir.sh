RUN_DIR=$1
task=$(echo "$2" | tr '[:lower:]' '[:upper:]')

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"

TASK_LIST=('LCP' 'LJP' 'LER' 'LRE' 'LAM')
if [[ ! " ${TASK_LIST[@]} " =~ " ${task} " ]]; then
  echo "Error task input ${Task}. Task value should be one of ${TASK_LIST[@]}"
  exit 1
fi

ALPHA_LIST=(0.1 1.0 10.0)
for alpha in ${ALPHA_LIST[@]}
do
  echo "run dir. 10 clients with ${alpha}"
  if [[ $task == "LRE" ]]; then
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/eval_f1_macro/dir_10_${alpha}.yaml"
  elif [[ $task == "LAM" ]]; then
    config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/eval_f1_micro/dir_10_${alpha}.yaml"
  else
    config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/dir_10_${alpha}.yaml"
  fi
  echo "config file in ${config_file}"
  bash fed_run.sh ${RUN_DIR} ${task} fedavg ${config_file} 18000 0 0 1
done
