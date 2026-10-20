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


class msrVTTDataset(Dataset):
    def __init__(self,
                 video_path: str,
                 height: int = 512,
                 width: int = 512,
                 n_sample_frames: int = 24,
                 dataset_set: str = "train",
                 ):
        self.semantic_bank = {
            "music": 0,
            "people": 1,
            "gaming": 2,
            "sports/actions": 3,
            "news/events/politics": 4,
            "education": 5,
            "tv shows": 6,
            "movie/comedy": 7,
            "animation": 8,
            "vehicles/autos": 9,
            "howto": 10,
            "travel": 11,
            "science/technology": 12,
            "animals/pets": 13,
            "kids/family": 14,
            "documentary": 15,
            "food/drink": 16,
            "cooking": 17,
            "beauty/fashion": 18,
            "advertisement": 19
        }
        self.data_dir = video_path
        self.dataset_set = dataset_set

        if self.dataset_set == "train":
            self.video_info_path = os.path.join(self.data_dir, "train_val_videodatainfo.json")
        else:
            self.video_info_path = os.path.join(self.data_dir, "test_videodatainfo.json")

        spatial_transform = 'resize_center_crop'

        video_length = n_sample_frames

        self.video_length = video_length
        self.resolution = [height, width]

        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(self.resolution)
            elif spatial_transform == "resize_center_crop":
                # assert (self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(self.resolution, antialias=True),
                    transforms_video.CenterCropVideo(self.resolution),
                ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

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
            """
            if self.dataset_set == "train":
                video_path = os.path.join(self.data_dir, "TrainValVideo/" + sample + '.mp4')
            else:
                video_path = os.path.join(self.data_dir, "TestVideo/" + sample + '.mp4')

            try:
                video_reader = VideoReader(video_path, ctx=cpu(0))
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            # average FPS(frame per second)
            fps_ori = video_reader.get_avg_fps()
            video_length = self.video_length

            # number of all frame (current video)
            allf = len(video_reader)

            all_frames = list(range(len(video_reader)))
            idx = int(len(all_frames) / 2)
            frame_indices = all_frames[idx - int(video_length / 2):idx + int(video_length / 2)]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}")
                index += 1
                continue

        assert (frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()

        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        assert (frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[
            1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        # float -> int
        frames = frames.byte()

        # shape (c,f,h,w) -> (f,c,h,w)
        frames = frames.permute(1, 0, 2, 3)
        frames = (frames / 127.5 - 1.0)
        """
        # example = {'pixel_values': frames, 'sentence': cur_video_sentence, 'video_id': sample}
        example = {'sentence': cur_video_sentence, 'video_id': sample}

        # return cur_video_info, cur_sentences_info["sen_id"].to_list(), example
        return example

    def __len__(self):
        with open(self.video_info_path, "r", encoding="utf-8") as f:
            video_info = json.load(f)
            dataset_len = len(video_info["videos"])
        return dataset_len


class PandaDataset(Dataset):
    def __init__(self,
                 video_path: str,
                 width: int = 512,
                 height: int = 512,
                 n_sample_frames: int = 24,
                 dataset_set: str = "train",
                 ):
        self.data_dir = video_path
        self.dataset_set = dataset_set

        self.length = n_sample_frames
        spatial_transform = 'resize_center_crop'

        if self.dataset_set == "train":
            self.video_info_path = os.path.join(self.data_dir, "panda70m_training.csv")
        else:
            self.video_info_path = os.path.join(self.data_dir, "panda70m_testing.csv")

        self.resolution = [height, width]

        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms_video.RandomCropVideo(self.resolution)
            elif spatial_transform == "resize_center_crop":
                # assert (self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(self.resolution, antialias=True),
                    transforms_video.CenterCropVideo(self.resolution),
                ])
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms_video.CenterCropVideo(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def __getitem__(self, index):
        videoID = ""
        """
        clip_number = 0
        vframes = []
        """

        with open(self.video_info_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            curr_line = 0
            for row in reader:
                if curr_line == index:
                    videoID = row["videoID"]
                    url = row["url"]
                    timestamp = row["timestamp"]
                    caption = eval(row['caption'])
                    matching_score = row["matching_score"]
                    desirable_filtering = row["desirable_filtering"]
                    shot_boundary_detection = row["shot_boundary_detection"]
                    """
                    numbers = len(eval(row['caption'])[0])
                    clip_number = numbers
                    """
                    """
                    for number in range(numbers):
                        if self.dataset_set == "train":
                            video_path = os.path.join(self.data_dir,
                                                      "TrainClipVideo/" + videoID + '_clip' + str(number) + ".mp4")
                        else:
                            video_path = os.path.join(self.data_dir,
                                                      "TestClipVideo/" + videoID + '_clip' + str(number) + ".mp4")

                        video_reader = VideoReader(video_path, ctx=cpu(0))
                        # average FPS(frame per second)
                        fps_ori = video_reader.get_avg_fps()
                        video_length = self.length

                        # number of all frame (current video)
                        allf = len(video_reader)
                        frame_indices = range(allf)
                        all_frames = list(range(len(video_reader)))
                        idx = int(len(all_frames) / 2)
                        frame_indices = all_frames[idx - int(video_length / 2):idx + int(video_length / 2)]
                        frames = video_reader.get_batch(frame_indices)

                        # shape (f,h,w,c) -> (c,f,h,w)
                        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()

                        if self.spatial_transform is not None:
                            frames = self.spatial_transform(frames)
                        assert (frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[
                            1]), f'frames={frames.shape}, self.resolution={self.resolution}'
                        # float -> int
                        frames = frames.byte()
                        # shape (c,f,h,w) -> (f,c,h,w)
                        frames = frames.permute(1, 0, 2, 3)
                        frames = (frames / 127.5 - 1.0)
                        vframes.append(frames)
                        curr_line = curr_line + 1
                        break
                    """
                curr_line = curr_line + 1

        """
        example = {'pixel_values': vframes, 'sentence': caption, 'videoID': videoID,
                   'clip_number': clip_number, }
        """
        # dataset cons
        """
        example = {
            'videoID': videoID,
            'url': url,
            'timestamp': timestamp,
            'sentence': caption,
            'matching_score': matching_score,
            'desirable_filtering': desirable_filtering,
            'shot_boundary_detection': shot_boundary_detection,
        }
        """
        example = {'videoID': videoID, 'sentence': caption}

        return example

    def __len__(self):
        with open(self.video_info_path, "r", encoding="utf-8") as csvfile:
            lines = csvfile.readlines()
            dataset_len = len(lines) - 1  # remove header
        return dataset_len


"""
folder = "/home/my/data/youtube"
Panda = PandaDataset(video_path=folder, n_sample_frames=23)
Panda.__getitem__(videoID="_1G2sz6ZzYs", number=0)
"""
