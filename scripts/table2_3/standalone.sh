RUN_DIR=$1
task=$(echo "$2" | tr '[:lower:]' '[:upper:]')

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"

TASK_LIST=('LCP' 'LJP' 'LER' 'LRE' 'LAM')
if [[ ! " ${TASK_LIST[@]} " =~ " ${task} " ]]; then
  echo "Error task input ${Task}. Task value should be one of ${TASK_LIST[@]}"
  exit 1
fi


if [[ $task == "LRE" ]]; then
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/eval_f1_macro/local.yaml"
elif [[ $task == "LAM" ]]; then
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/eval_f1_micro/local.yaml"
else
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/local.yaml"
fi
echo "config file in ${config_file}"


bash cen_run.sh ${RUN_DIR} ${task} local ${config_file} 18000 0
