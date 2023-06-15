RUN_DIR=$1
task=$2

cd tools/legal_scripts

python part_sampled_util.py --run_dir ${RUN_DIR} --task ${task}
