RUN_DIR=$1

# default in fed_silo_msgd.yaml:
# fed_opt_type: fedmsgd
# m_t: 0.92
# eta: 1.0

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
  config_file="config_files/${task,,}/eval_f1_macro/fed_silo_msgd.yaml"
elif [[ $task == "LAM" ]]; then
  config_file="config_files/${task,,}/eval_f1_micro/fed_silo_msgd.yaml"
else
  config_file="config_files/${task,,}/fed_silo.yaml"
fi
echo "config file in ${config_file}"


bash fed_run.sh ${RUN_DIR} ${task} fedopt ${config_file} 18000 0,0,1


# SEARCH PARAMETERS
# python fed_sweep_once.py LCP fedopt config_files/lcp/fed_silo_msgd.yaml fedopt_msgd_fine-tuning 18000 0,0,1

# python fed_sweep_once.py LJP fedopt config_files/ljp/fed_silo_msgd.yaml fedopt_msgd_fine-tuning 18000 0,0,1

# python fed_sweep_once.py LER fedopt config_files/ler/fed_silo_msgd.yaml fedopt_msgd_fine-tuning 18000 0,0,1

# python fed_sweep_once.py LRE fedopt config_files/lre/fed_silo_msgd.yaml fedopt_msgd_fine-tuning 18000 0,0,1

# python fed_sweep_once.py LCP fedopt config_files/lcp/fed_silo_msgd.yaml LAM_fedopt_msgd_fine-tuning 18000 0,0,1
