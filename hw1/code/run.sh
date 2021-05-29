source /zhuchenyang/.bashrc
export PYTHONPATH=/zhuchenyang/source/cv/meg_hw1:$PYTHONPATH
export MEGENGINE_LOGGING_LEVEL=ERROR;
python tools/train.py -f configs/fcos_cfg.py -d /zhuchenyang/Dataset -n 1 -l ./train_log/fcos
