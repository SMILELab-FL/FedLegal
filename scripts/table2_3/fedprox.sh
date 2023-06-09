RUN_DIR=$1
task=$(echo "$2" | tr '[:lower:]' '[:upper:]')

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"

TASK_LIST=('LCP' 'LJP' 'LER' 'LRE' 'LAM')
if [[ ! " ${TASK_LIST[@]} " =~ " ${task} " ]]; then
  echo "Error task input ${Task}. Task value should be one of ${TASK_LIST[@]}"
  exit 1
fi

# default 0.005 mu in fed_silo.yaml
if [[ $task == "LRE" ]]; then
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/eval_f1_macro/fed_silo.yaml"
elif [[ $task == "LAM" ]]; then
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/eval_f1_micro/fed_silo.yaml"
else
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/fed_silo.yaml"
fi
echo "config file in ${config_file}"


bash fed_run.sh ${RUN_DIR} ${task} fedprox ${config_file} 18000 0 0 1

# SEARCH PARAMETERS
# python fed_sweep_once.py LCP fedprox config_files/lcp/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LJP fedprox config_files/ljp/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LER fedprox config_files/ler/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LRE fedprox config_files/lre/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LCP fedprox config_files/lcp/fed_silo.yaml LAM_mu_fine-tuning 18000 0 0 1
