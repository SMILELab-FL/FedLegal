RUN_DIR=$1
task=LCP

bash cen_run_sampled.sh ${RUN_DIR} ${task} local config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/local.yaml 18000 0.1 0

bash cen_run_sampled.sh ${RUN_DIR} ${task} local config_files/$(echo "$task" | tr '[:upper:]' '[:lower:]')/local.yaml 18000 0.5 0

