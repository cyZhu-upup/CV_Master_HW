# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import bisect
import copy
import os
import time

import megengine as mge
import megengine.distributed as dist
from megengine.autodiff import GradManager
from megengine.data import DataLoader, Infinite, RandomSampler
from megengine.data import transform as T
from megengine.optimizer import SGD

from tools.data_mapper import data_mapper
from tools.utils import (
    AverageMeter,
    DetectionPadCollator,
    GroupedRandomSampler,
    get_config_info,
    import_from_file
)

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", default="net.py", type=str, help="net description file" # configs/faster_rcnn_res50_800size_fsdet_demo.py
    )
    parser.add_argument(
        "-w", "--weight_file", default=None, type=str, help="weights file",
    )
    parser.add_argument(
        "-n", "--devices", default=1, type=int, help="total number of gpus for training",
    )
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="batch size for training",
    )
    parser.add_argument(
        "-d", "--dataset_dir", default="/zhuchenyang/Dataset/", type=str,
    )

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # ------------------------ begin training -------------------------- #
    logger.info("Device Count = %d", args.devices)

    log_dir = "logs/{}".format(os.path.basename(args.file).split(".")[0])
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if args.devices > 1:
        trainer = dist.launcher(worker, n_gpus=args.devices)
        trainer(args)
    else:
        worker(args)


def worker(args):
    current_network = import_from_file(args.file)
    # 相当于import configs/faster_rcnn_res50_800size_fsdet_demo
    model = current_network.Net(current_network.Cfg())
    model.train()

    if dist.get_rank() == 0:
        logger.info(get_config_info(model.cfg))
        logger.info(repr(model))

    params_with_grad = [] # 需要用来计算梯度的向量
    for name, param in model.named_parameters():
        if "bottom_up.conv1" in name and model.cfg.backbone_freeze_at >= 1:
            continue
        if "bottom_up.layer1" in name and model.cfg.backbone_freeze_at >= 2: # 默认为2
            continue
        params_with_grad.append(param) # 冻结ResNet的第一层和第一个block

    opt = SGD(
        params_with_grad,
        lr=model.cfg.basic_lr * args.batch_size,
        momentum=model.cfg.momentum,
        weight_decay=model.cfg.weight_decay * dist.get_world_size(),
    )

    gm = GradManager()
    if dist.get_world_size() > 1:
        gm.attach(
            params_with_grad,
            callbacks=[dist.make_allreduce_cb("SUM", dist.WORLD)]
        )
    else:
        gm.attach(params_with_grad) # 用户如果希望后续计算张量的梯度，则需要在该张量上调用 attach 方法

    if args.weight_file is not None:
        # model.backbone.bottom_up.load_state_dict(weights, strict=False)
        logger.info("Loading Base-Pretrain weights...")
        weights = mge.load(args.weight_file)
        weight_new = {k:v for k, v in weights.items() if 'pred_' not in k} # 不传入rcnn的最后两层参数
        model.load_state_dict(weight_new, strict=False) #  strict为True ，state_dict 的键则必须与 state_dict 返回的键准确匹配

    if dist.get_world_size() > 1:
        dist.bcast_list_(model.parameters(), dist.WORLD)  # sync parameters

    if dist.get_rank() == 0:
        logger.info("Prepare dataset")
    train_loader = iter(build_dataloader(args.batch_size, args.dataset_dir, model.cfg))

    for epoch in range(model.cfg.max_epoch):
        train_one_epoch(model, train_loader, opt, gm, epoch, args)
        if dist.get_rank() == 0 : # and (epoch % 2 == 0 or epoch == model.cfg.max_epoch - 1)
            save_path = "logs/{}/epoch_{}.pkl".format(
                os.path.basename(args.file).split(".")[0], epoch
            )
            mge.save(
                {"epoch": epoch, "state_dict": model.state_dict()}, save_path,
            )
            logger.info("dump weights to %s", save_path)
            
        

def train_one_epoch(model, data_queue, opt, gm, epoch, args):
    def train_func(image, im_info, gt_boxes):
        with gm:
            loss_dict = model(image=image, im_info=im_info, gt_boxes=gt_boxes)
            gm.backward(loss_dict["total_loss"])
            loss_list = list(loss_dict.values())
        opt.step().clear_grad()
        return loss_list

    meter = AverageMeter(record_len=model.cfg.num_losses)
    time_meter = AverageMeter(record_len=2)
    log_interval = model.cfg.log_interval
    tot_step = model.cfg.nr_images_epoch // (args.batch_size * dist.get_world_size())
    for step in range(tot_step):
        adjust_learning_rate(opt, epoch, step, model.cfg, args)

        data_tik = time.time()
        mini_batch = next(data_queue)
        data_tok = time.time()

        tik = time.time()
        loss_list = train_func(
            image=mge.tensor(mini_batch["data"]),
            im_info=mge.tensor(mini_batch["im_info"]),
            gt_boxes=mge.tensor(mini_batch["gt_boxes"])
        )
        tok = time.time()

        time_meter.update([tok - tik, data_tok - data_tik])

        if dist.get_rank() == 0:
            info_str = "e%d, %d/%d, lr:%f, "
            loss_str = ", ".join(
                ["{}:%f".format(loss) for loss in model.cfg.losses_keys]
            )
            time_str = ", train_time:%.3fs, data_time:%.3fs"
            log_info_str = info_str + loss_str + time_str
            meter.update([loss.numpy() for loss in loss_list])
            if step % log_interval == 0:
                logger.info(
                    log_info_str,
                    epoch,
                    step,
                    tot_step,
                    opt.param_groups[0]["lr"],
                    *meter.average(),
                    *time_meter.average()
                )
                meter.reset()
                time_meter.reset()


def adjust_learning_rate(optimizer, epoch, step, cfg, args):
    base_lr = (
        cfg.basic_lr * args.batch_size * (
            cfg.lr_decay_rate
            ** bisect.bisect_right(cfg.lr_decay_stages, epoch)
        )
    )
    # Warm up
    lr_factor = 1.0
    if epoch == 0 and step < cfg.warm_iters:
        lr_factor = (step + 1.0) / cfg.warm_iters
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * lr_factor


def build_dataset(dataset_dir, cfg):
    data_cfg = copy.deepcopy(cfg.train_dataset)
    data_name = data_cfg.pop("name")

    data_cfg["root"] = os.path.join(dataset_dir, data_name, data_cfg["root"])

    if "ann_file" in data_cfg:
        data_cfg["ann_file"] = os.path.join(dataset_dir, data_name, data_cfg["ann_file"])

    data_cfg["order"] = ["image", "boxes", "boxes_category", "info"]

    return data_mapper[data_name](**data_cfg)


# pylint: disable=dangerous-default-value
def build_sampler(train_dataset, batch_size, aspect_grouping=[1]):
    def _compute_aspect_ratios(dataset):
        aspect_ratios = []
        for i in range(len(dataset)):
            info = dataset.get_img_info(i)
            aspect_ratios.append(info["height"] / info["width"])
        return aspect_ratios

    def _quantize(x, bins):
        return list(map(lambda y: bisect.bisect_right(sorted(bins), y), x))

    if len(aspect_grouping) == 0:
        return Infinite(RandomSampler(train_dataset, batch_size, drop_last=True))

    aspect_ratios = _compute_aspect_ratios(train_dataset)
    group_ids = _quantize(aspect_ratios, aspect_grouping)
    return Infinite(GroupedRandomSampler(train_dataset, batch_size, group_ids))


def build_dataloader(batch_size, dataset_dir, cfg):
    train_dataset = build_dataset(dataset_dir, cfg)
    train_sampler = build_sampler(train_dataset, batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        transform=T.Compose(
            transforms=[
                T.ShortestEdgeResize(
                    cfg.train_image_short_size,
                    cfg.train_image_max_size,
                    sample_style="choice",
                ),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ],
            order=["image", "boxes", "boxes_category"],
        ),
        collator=DetectionPadCollator(),
        num_workers=2,
    )
    return train_dataloader


if __name__ == "__main__":
    main()
