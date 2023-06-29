RUN_DIR=$1
task=LAM

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"


else
  config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/cen.yaml"
fi
echo "config file in ${config_file}"

echo "Layer 3 for centralized"
config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/layers_f1_micro/cen_lr_5e-5_ep_20_layers_3.yaml"
bash cen_run.sh ${RUN_DIR} ${task} centralized ${config_file} 18000 0

echo "Layer 6 for centralized"
config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/layers_f1_micro/cen_lr_5e-5_ep_20_layers_6.yaml"
bash cen_run.sh ${RUN_DIR} ${task} centralized ${config_file} 18000 0


echo "Layer 3 for standalone"
config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/layers_f1_micro/local_lr_5e-5_ep_20_layers_3.yaml"
bash cen_run.sh ${RUN_DIR} ${task} local ${config_file} 18000 0

echo "Layer 6 for standalone"
config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/layers_f1_micro/local_lr_5e-5_ep_20_layers_6.yaml"
bash cen_run.sh ${RUN_DIR} ${task} local ${config_file} 18000 0


echo "Layer 3 for fedavg"
config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/layers_f1_micro/fed_silo_lr_5e-5_layers_6.yaml"
bash cen_run.sh ${RUN_DIR} ${task} fedavg ${config_file} 18000 0 0 1

echo "Layer 6 for fedavg"
config_file="config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/layers_f1_micro/fed_silo_lr_5e-5_layers_6.yaml"
bash cen_run.sh ${RUN_DIR} ${task} fedavg ${config_file} 18000 0 0 1

# The results of layer 12 are the same with results in Table2
