RUN_DIR=$1
task=$(echo "$2" | tr '[:lower:]' '[:upper:]')
cluster=$3  # empty for False

echo "RUN_DIR: ${RUN_DIR}"
echo "task: ${task}"
echo "cluster: ${cluster}"

TASK_LIST=('LCP' 'LJP' 'LER' 'LRE' 'LAM')
if [[ ! " ${TASK_LIST[@]} " =~ " ${task} " ]]; then
  echo "Error task input ${Task}. Task value should be one of ${TASK_LIST[@]}"
  exit 1
fi

cd tools/legal_scripts
if [[ $cluster ]]; then
  python legal.py --run_dir ${RUN_DIR} --task ${task} --cluster
else
  python legal.py --run_dir ${RUN_DIR} --task ${task}
fi

