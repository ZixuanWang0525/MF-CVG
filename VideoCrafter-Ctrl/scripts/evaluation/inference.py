import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import repeat
from einops import rearrange, repeat
from functools import partial
import torch
from open_clip.tokenizer import _tokenizer as text_tokenizer
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from funcs import batch_ddim_sampling
from utils.utils import instantiate_from_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from dataset.dataloader import msrVTTDataset, PandaDataset
from dataset.comploader import msrVTTCompDataset, PandaCompDataset

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "A&R"))
from ChatGPT import prompt_bbox_process

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "boxdiff"))
from ptp_utils import AttentionStore, register_control, DiffusionLatentStore

from torch.utils.data import DataLoader
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    # 20230211
    # B&D, 20112023
    # R&B, 23022011
    # A&R, 11023202
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt", )
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="eta for ddim sampling (0.0 yields deterministic sampling)", )
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None,
                        help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    ## for compositional T2V
    # parser.add_argument("--dataset_name", type=str, default="panda70m", help=None)
    parser.add_argument("--dataset_name", type=str, default="msrVTT", help=None)
    return parser


def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    # data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 1.5: register controller
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    attention_store = AttentionStore(temporal_length=frames)
    diffusion_latent_store = DiffusionLatentStore()
    register_control(model, frames, attention_store, diffusion_latent_store)

    ## step 2: load data
    ## -----------------------------------------------------------------
    dataset_name = args.dataset_name
    dataset = None
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    if dataset_name == 'msrVTT':
        msrVTT_video_path = os.path.join(proj_dir, "dataset", "msrvtt_compositional")
        dataset = msrVTTCompDataset(
            video_path=msrVTT_video_path,
            dataset_set="test"
        )
    elif dataset_name == "panda70m":
        panda70m_video_path = os.path.join(proj_dir, "dataset", "panda70m_compositional")
        dataset = PandaCompDataset(
            video_path=panda70m_video_path,
            dataset_set="test"
        )
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    prompt_list_rank = []
    filename_list_rank = []
    guidance_signal_list_rank = []
    if dataset_name == 'msrVTT':
        for example in dataloader:
            sentence, videoID, scene_graph, video_planning, new_sentence = \
                example["sentence"], example["video_id"], example["scene_graph"], example["video_planning"], example[
                    "new_sentence"]
            filename = videoID[0]
            if not os.path.exists(os.path.join(args.savedir, filename + ".mp4")):
                o_names, position, box_att = prompt_bbox_process(new_sentence[0], video_planning, args.height,
                                                                 args.width, tokenizer=text_tokenizer)
                prompt_list_rank.append(new_sentence[0])
                filename_list_rank.append(filename)
                guidance_signal_list_rank.append(
                    dict(
                        boxes=box_att,
                        object_position=position,
                        o_names=o_names
                    )
                )
            else:
                print(filename + " already have !")

    elif dataset_name == "panda70m":
        for example in dataloader:
            videoID, sentences, scene_graphs, video_plannings, new_sentences, clip_numbers = \
                example["videoID"], example["sentence"], example["scene_graph"], example["video_planning"], example[
                    "new_sentence"], example["clip_number"]
            for index in range(len(new_sentences)):
                filename = videoID[0] + "_" + "clip_" + str(clip_numbers[index].item())
                if not os.path.exists(os.path.join(args.savedir, filename + ".mp4")):
                    o_names, position, box_att = prompt_bbox_process(new_sentences[index][0], video_plannings[index],
                                                                     args.height, args.width, tokenizer=text_tokenizer)
                    prompt_list_rank.append(new_sentences[index][0])
                    filename_list_rank.append(filename)
                    guidance_signal_list_rank.append(
                        dict(
                            boxes=box_att,
                            object_position=position,
                            o_names=o_names
                        )
                    )
                else:
                    print(filename + " already have !")

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds + 1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in range(0, n_rounds):
        print("current_index", idx)
        #
        # if idx == 250:
        # if idx == 150:
            # break
        print(f'[rank:{gpu_no}] batch-{idx + 1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx * args.bs
        idx_e = min(idx_s + args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps] * batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        # guidance
        # compositional_guidance_signals = guidance_signal_list_rank[idx_s:idx_e]
        compositional_guidance_signals = {
            "frame_number": frames,
            "attention_store": attention_store,
            "diffusion_latent_store": diffusion_latent_store,
            "guidance_signal": guidance_signal_list_rank[idx_s:idx_e]
        }
        if isinstance(prompts, str):
            prompts = [prompts]
        # prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps}
        else:
            raise NotImplementedError

        ## inference
        print("Now" + str(filenames) + "inference !")
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, compositional_guidance_signals, args.n_samples,
                                            args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, **kwargs)
        ## b,samples,c,t,h,w
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s" % now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)
