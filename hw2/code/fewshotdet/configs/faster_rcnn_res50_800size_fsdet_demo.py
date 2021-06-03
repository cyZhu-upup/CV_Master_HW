# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import models


class CustomerConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="mini365",
            root="images",
            ann_file="tasks/train_30shot.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="mini365",
            root="images",
            ann_file="tasks/test.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 20

        # ------------------------ training cfg ---------------------- #
        self.basic_lr = 0.02 / 8
        #self.max_epoch = 24
        #self.lr_decay_stages = [16, 21]
        self.max_epoch = 36
        self.lr_decay_stages = [28, 34]
        self.nr_images_epoch = 400
        self.warm_iters = 100
        self.log_interval = 10


Net = models.FasterRCNN
Cfg = CustomerConfig
