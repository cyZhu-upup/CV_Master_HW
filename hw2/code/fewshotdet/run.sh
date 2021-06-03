source /zhuchenyang/.bashrc

export PYTHONPATH=./:$PYTHONPATH
export MEGENGINE_LOGGING_LEVEL=ERROR
python3 tools/train.py -n 1 -b 2 -f configs/faster_rcnn_res50_800size_fsdet_demo.py  -w weights/faster_rcnn_res101_coco_3x_800size_42dot6_2538b0ff.pkl

#python3 tools/test.py -n 1 -se 1 -f configs/faster_rcnn_res50_800size_fsdet_demo.py 