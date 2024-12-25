# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import OccModel
from ultralytics.utils import DEFAULT_CFG, RANK, checks
from ultralytics.utils.torch_utils import de_parallel
from torchvision import models
import torch
from .modules.depth_model import Depth_Model
from .modules.embedding import Embedding_Module
from .modules.pcdnet import PcdNet
from .modules.spnet import SPNet

class OccTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Occ Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal="lidar"
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OccModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        model.dfe = Depth_Model()
        model.pcdfe = PcdNet()
        model.spfe = SPNet()
        model.dspfe = SPNet()
        model.embed = Embedding_Module(in_im=[72,72,72],in_de=[512,1024,2048],in_sp=[216,576,1512],in_dsp=[216,576,1512],hidden_ch=256,out_ch=72)
        
        return model

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images for YOLOOcc training, adjusting formatting and dimensions as needed."""
        batch = super().preprocess_batch(batch)
        bs = self.batch_size
        max_depth = batch["depth"].view(bs, -1).max(dim=-1).values
        batch["depth"] = (batch["depth"].float() / max_depth.view(-1, 1, 1, 1)).to(self.device, non_blocking=True)
        # TODO: @VE process of pcd
        # import ipdb;ipdb.set_trace()
        max_uvz = batch["pcd"].view(-1,3).max(dim=0).values
        batch["pcd"] = (batch["pcd"].float() / max_uvz.view(1,1,-1)).to(self.device, non_blocking=True)
        batch["spixel"] = batch["spixel"].to(self.device, non_blocking=True).float() / batch["spixel"].max().item()
        batch["dspixel"] = batch["dspixel"].to(self.device, non_blocking=True).float() / batch["dspixel"].max().item()
        return batch