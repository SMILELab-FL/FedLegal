RUN_DIR=$1
task=${2^^}

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"

TASK_LIST=('LCP' 'LJP' 'LER' 'LRE' 'LAM')
if [[ ! " ${TASK_LIST[@]} " =~ " ${task} " ]]; then
  echo "Error task input ${Task}. Task value should be one of ${TASK_LIST[@]}"
  exit 1
fi

# default 0.005 mu in fed_silo.yaml
if [[ $task == "LRE" ]]; then
  config_file="config_files/${task,,}/eval_f1_macro/fed_silo.yaml"
elif [[ $task == "LAM" ]]; then
  config_file="config_files/${task,,}/eval_f1_micro/fed_silo.yaml"
else
  config_file="config_files/${task,,}/fed_silo.yaml"
fi
echo "config file in ${config_file}"


bash fed_run.sh ${RUN_DIR} ${task} ditto ${config_file} 18000 0 0 1

# SEARCH PARAMETERS
# python fed_sweep_once.py LCP ditto config_files/lcp/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LJP ditto config_files/ljp/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LER ditto config_files/ler/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LRE ditto config_files/lre/fed_silo.yaml mu_fine-tuning 18000 0 0 1

# python fed_sweep_once.py LCP ditto config_files/lcp/fed_silo.yaml LAM_mu_fine-tuning 18000 0 0 1
