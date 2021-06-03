source /zhuchenyang/.bashrc

export PYTHONPATH=./:$PYTHONPATH
export MEGENGINE_LOGGING_LEVEL=ERROR


python3 tools/test.py -n 1 -se 0 -ee 35 -f configs/faster_rcnn_res50_800size_fsdet_demo.py 
