import math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import default_collate
from pathlib import Path
import tqdm
from einops import rearrange

from meshgen.util import instantiate_from_config
from diffusers.training_utils import EMAModel
from meshgen.utils.io import load_mesh, write_video
from meshgen.utils.render import render_mesh_spiral_offscreen
from meshgen.modules.timm import trunc_normal_, Mlp


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class BaseModel(pl.LightningModule):
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def log_prefix(self, loss_dict, prefix):
        for k, v in loss_dict.items():
            self.log(
                f"{prefix}/{k}",
                v,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                batch_size=self.trainer.train_dataloader.loaders.batch_size,
            )

    def shared_step(self, batch, batch_idx):
        raise NotImplementedError("shared_step must be implemented in subclass")

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_prefix(loss_dict, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_prefix(loss_dict, "val")
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        # TODO: remove weight decay for some layers
        param_groups = self.get_optim_groups()
        # opt = torch.optim.AdamW(
        #     self.parameters(), lr=lr, weight_decay=self.weight_decay
        # )
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=self.weight_decay)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def on_train_epoch_end(self, *args, **kwargs):
        self.eval()
        logdir = self.trainer.logdir
        this_logdir = Path(logdir) / f"spirals/epoch_{self.current_epoch}"
        this_logdir.mkdir(exist_ok=True, parents=True)
        self.test_in_the_wild(this_logdir)
        self.train()

    def on_train_start(self):
        return
        self.eval()
        logdir = self.trainer.logdir
        this_logdir = Path(logdir) / f"spirals/before_training"
        this_logdir.mkdir(exist_ok=True, parents=True)
        self.test_in_the_wild(this_logdir)
        self.train()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.eval()
        if self.vis_every is None:
            return

        # if (self.global_step + 1) % self.vis_every != 0 and self.global_step != 0:
        #     return
        if (self.global_step + 1) % self.vis_every != 0:
            return

        logdir = self.trainer.logdir  ###
        this_logdir = Path(logdir) / f"spirals/step_{self.global_step}"
        this_logdir.mkdir(exist_ok=True, parents=True)
        self.test_in_the_wild(this_logdir)
        self.train()

    def test_in_the_wild(self, save_dir):
        raise NotImplementedError("test_in_the_wild must be implemented in subclass")

    def get_optim_groups(self):
        raise NotImplementedError("get_optim_groups must be implemented in subclass")
