from torch.utils.data import Dataset
from torchvision import transforms
from decord import VideoReader, cpu

import csv
import os
import json
import random
import torch
import torchvision.transforms._transforms_video as transforms_video

import pandas as pd


class msrVTTCompDataset(Dataset):
    def __init__(self,
                 video_path: str,
                 height: int = 512,
                 width: int = 512,
                 n_sample_frames: int = 24,
                 dataset_set: str = "train",
                 ):
        self.data_dir = video_path
        self.dataset_set = dataset_set

        if self.dataset_set == "train":
            self.video_info_path = os.path.join(self.data_dir, "train_val_videodatainfo.json")
        else:
            self.video_info_path = os.path.join(self.data_dir, "test_videodatainfo.json")

        self.planning_dir = self.data_dir.split("_")[0] + "_video_planning"
        # self.planning_dir = self.data_dir.split("_")[0] + "_video_planning_1"
        print(self.planning_dir)

    def __getitem__(self, index):
        while True:
            with open(self.video_info_path, "r", encoding="utf-8") as f:
                video_info = json.load(f)
                cur_video_info = video_info["videos"][index]
                # sample = video_info["videos"][index]["video_id"]
                sample = cur_video_info["video_id"]

                sentences = pd.DataFrame(video_info["sentences"])
                cur_sentences_info = sentences.loc[sentences["video_id"] == sample]
                # cur_video_sentences = sentences.loc[sentences["video_id"] == sample][["caption"]]
                cur_video_sentences = cur_sentences_info[["caption"]]
                caption = cur_video_sentences[["caption"]].values.tolist()
                """
                sentence_id = random.randint(0, len(caption) - 1)
                cur_video_sentence = caption[sentence_id]
                """
                cur_video_sentence = caption
                break  ###

        scene_graph = None
        video_planning = None
        new_sentence = None
        # scene graph
        with open(os.path.join(self.planning_dir, sample + "scene_graph.json"), "r",
                  encoding="utf-8") as scene_graph_file:
            scene_graph = json.load(scene_graph_file)
        # video planning
        with open(os.path.join(self.planning_dir, sample + "video_planning.json"), "r",
                  encoding="utf-8") as video_planning_file:
            video_planning = json.load(video_planning_file)
        # new sentence
        with open(os.path.join(self.planning_dir, sample + "new_sentence.json"), "r",
                  encoding="utf-8") as new_sentence_file:
            new_sentence = json.load(new_sentence_file)

        # example = {'pixel_values': frames, 'sentence': cur_video_sentence, 'video_id': sample}
        example = {'sentence': cur_video_sentence, 'video_id': sample,
                   "scene_graph": scene_graph, 'video_planning': video_planning, "new_sentence": new_sentence}

        # return cur_video_info, cur_sentences_info["sen_id"].to_list(), example
        return example

    def __len__(self):
        with open(self.video_info_path, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            dataset_len = len(video_info["videos"])
        return dataset_len


class PandaCompDataset(Dataset):
    def __init__(self,
                 video_path: str,
                 # width: int = 512,
                 # height: int = 512,
                 # n_sample_frames: int = 24,
                 dataset_set: str = "train",
                 ):
        # only need caption, no video frame
        self.orig_dir = video_path
        self.dataset_set = dataset_set

        if self.dataset_set == "train":
            self.video_info_path = os.path.join(self.orig_dir, "panda70m_training.csv")
        else:
            self.video_info_path = os.path.join(self.orig_dir, "panda70m_testing.csv")

        self.planning_dir = self.orig_dir.split("_")[0] + "_video_planning"
        # self.planning_dir = self.orig_dir.split("_")[0] + "_video_planning_4"
        print(self.planning_dir)

    def __getitem__(self, index):
        videoID = ""
        # 1 video, 3 caption
        caption_list = []
        scene_graph_list = []
        video_planning_list = []
        new_sentence_list = []
        clip_number_list = []
        print(index)

        with open(self.video_info_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            curr_line = 0
            for row in reader:
                if curr_line == index:
                    videoID = row["videoID"]
                    captions = eval(row['caption'])
                    for i in range(len(captions)):
                        if captions[i] == "-":
                            pass
                        else:
                            caption_list.append(captions[i])
                            # scene graph
                            with open(os.path.join(self.planning_dir, videoID
                                                                      + "_" + "clip" + str(i) + "_scene_graph.json"),
                                      "r", encoding="utf-8") as scene_graph_file:
                                scene_graph_list.append(json.load(scene_graph_file))
                            # video planning
                            with open(os.path.join(self.planning_dir, videoID
                                                                      + "_" + "clip" + str(i) + "_video_planning.json"),
                                      "r", encoding="utf-8") as video_planning_file:
                                video_planning_list.append(json.load(video_planning_file))
                            # new sentence
                            with open(os.path.join(self.planning_dir, videoID
                                                                      + "_" + "clip" + str(i) + "_new_sentence.json"),
                                      "r", encoding="utf-8") as new_sentence_file:
                                new_sentence_list.append(json.load(new_sentence_file))
                            clip_number_list.append(i)
                curr_line = curr_line + 1

        # n sentences
        print(videoID)
        example = {'videoID': videoID, 'sentence': caption_list, 'scene_graph': scene_graph_list,
                   'video_planning': video_planning_list, 'new_sentence': new_sentence_list,
                   "clip_number": clip_number_list}

        return example

    def __len__(self):
        with open(self.video_info_path, "r", encoding="utf-8") as csvfile:
            lines = csvfile.readlines()
            dataset_len = len(lines) - 1  # remove header
        return dataset_len


# pcd = PandaCompDataset(video_path="./panda70m_compositional", dataset_set="test")
# pcd.__getitem__(index=1)

# mcd = msrVTTCompDataset(video_path="./msrvtt_compositional", dataset_set="test")
# mcd.__getitem__(index=1)
