import os
import argparse
import datetime
import numpy as np
import torch
import json
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import torch.distributed as dist

from meshgen.util import instantiate_from_config
from meshgen.utils.hf_weights import (
    pbr_decomposer_path,
    texture_inpainter_path,
    mv_generator_path,
)
from meshgen.utils.misc import set_if_none


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["configs/texgen.yaml"],
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--no_ignore_alpha",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs/texturing",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "-m",
        "--meta",
        type=str,
        default=None,
        help="The meta data file for all meshes an their corresponding images",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="The meta data file for all meshes an their corresponding images",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--default_logger",
        type=str,
        help="The default logger to use",
        choices=["testtube", "wandb"],
        default="wandb",
    )
    return parser


def main():
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    seed_everything(opt.seed)

    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""

    if local_rank == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    else:
        now = None

    dist.barrier()
    outputs = [now]

    dist.broadcast_object_list(outputs, 0)
    now = outputs[0]

    nowname = now + name + opt.postfix

    include_ema = False

    if opt.base == []:
        opt.base = ["configs/texgen.yaml"]
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    logdir = Path(opt.logdir) / nowname
    config.params.exp_dir = str(logdir)
    if local_rank == 0:
        logdir.mkdir(parents=True, exist_ok=True)

    cfgdir = logdir / "configs"
    if local_rank == 0:
        cfgdir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, cfgdir / "config.yaml")

    meta = json.load(open(opt.meta, "r"))
    max_items = opt.max_items
    if max_items < 0:
        max_items = len(meta)

    if not include_ema:
        new_meta = []
        for m in meta:
            if "ema" not in m["mesh"]:
                new_meta.append(m)
        meta = new_meta

    # meta = sorted(meta, key=lambda x: x["mesh"])[local_rank::world_size]
    meta = meta[local_rank::world_size]

    set_if_none(
        config.params.multiview_generator.params, "ckpt_path", mv_generator_path
    )
    set_if_none(
        config.params.texture_inpainter.params,
        "ckpt_path",
        texture_inpainter_path,
    )
    set_if_none(config.params.pbr_decomposer.params, "ckpt_path", pbr_decomposer_path)

    print(f"Using config file: {opt.base}")
    print(f"Using meta file: {opt.meta}")
    print(f"Using multi-view generator: {mv_generator_path}")
    print(f"Using texture inpainter: {texture_inpainter_path}")
    print(f"Using PBR decomposer: {pbr_decomposer_path}")

    model = instantiate_from_config(config)
    for idx, m in enumerate(meta):
        if idx >= max_items:
            break
        try:
            model(
                m["mesh"],
                m["image"],
                verbose=True,
                front_view_only=False,
                skip_front_view=True,
                debug=True,
                ignore_alpha=not opt.no_ignore_alpha,
            )
        except:
            raise
            print(f"Error on {m['mesh']}")
            pass


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
