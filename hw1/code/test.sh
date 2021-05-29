source /zhuchenyang/.bashrc
export PYTHONPATH=/zhuchenyang/source/cv/meg_hw1:$PYTHONPATH
export MEGENGINE_LOGGING_LEVEL=ERROR
python tools/test.py -f configs/fcos_cfg.py -d /zhuchenyang/Dataset -n 1 -se 26 -ee 26  -w train_log/fcos/