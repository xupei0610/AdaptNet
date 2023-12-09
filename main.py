import os
import importlib

import env_adapt as env
from model_adapt import ACModel, Discriminator, AdaptNet, MapCNN

import torch
import numpy as np
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--meta", type=str, default=None,
    help="Pretrained meta checkpoint file.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
settings = parser.parse_args()

    
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def test(env, model):
    model.eval()
    env.reset()
    while not env.request_quit:
        obs, info = env.reset_done()
        seq_len = info["ob_seq_lens"]
        if "map" in info:
            m = info["map"]
            obs = torch.cat((obs, model.critic.map(m)), -1)
            model.actor.g = model.actor.map(m)
        actions = model.act(obs, seq_len-1)
        env.step(actions)

if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }
    else:
        discriminators = {"_/full": env.DiscriminatorConfig()}
    env_cls = getattr(env, config.env_cls)
    if not hasattr(config, "env_params"):
        setattr(config, "env_params", {})

    env = env_cls(1,
        discriminators=discriminators,
        compute_device=settings.device, 
        **config.env_params
    )
    env.episode_length = 500000
    map_dim = 256 if hasattr(env, "info") and "map" in env.info else 0



    value_dim = len(env.discriminators)+env.rew_dim
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim, meta_goal_dim=map_dim)
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    if settings.meta is not None and os.path.exists(settings.meta):
        if os.path.isdir(settings.meta):
            ckpt = os.path.join(settings.meta, "ckpt")
        else:
            ckpt = settings.meta
            settings.meta = os.path.dirname(ckpt)
        if os.path.exists(ckpt):
            print("Load meta-model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=device)
            pretrained = dict()
            for k, p in state_dict["model"].items():
                if "actor" in k or "actor_ob_normalizer" in k:
                    pretrained[k] = p
            model.load_state_dict(pretrained, strict=False)
    model.discriminators = discriminators

    model.actor = AdaptNet(model, g_dim=map_dim)
    if "map" in env.info:
        model.critic.map = MapCNN()
        model.actor.map = MapCNN()
    model.to(device)

    if settings.ckpt is not None and os.path.exists(settings.ckpt):
        if os.path.isdir(settings.ckpt):
            ckpt = os.path.join(settings.ckpt, "ckpt")
        else:
            ckpt = settings.ckpt
            settings.ckpt = os.path.dirname(ckpt)
        if os.path.exists(ckpt):
            print("Load model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
            model.load_state_dict(state_dict["model"])
    env.render()
    test(env, model)
