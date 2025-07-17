import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import cv2
import json
import matplotlib.pyplot as plt
import re
from typing import List, Dict


class ObjectMotionDataset(Dataset):
    def __init__(self, dataset_dir, video_paths_json, num_frames=3, 
                 augment_bg = False, resize_to = None, 
                 augment_static = False, augment_reverse = False,
                 has_brdf = True, static_prob = 0.2):
        
        self.dataset_dir = dataset_dir
        self.augment_bg = augment_bg
        self.video_paths_json = video_paths_json
        self.resize_to = resize_to
        self.num_frames = num_frames
        self.augment_static = augment_static # make some videos static for training
        self.augment_reverse = augment_reverse # reverse video sequence augmentation
        self.has_brdf = has_brdf
        self.samples = self._concat_paths()
        self.static_prob = static_prob
    
    def _concat_paths(self):
        with open(self.video_paths_json, "r") as f:
            video_paths = json.load(f) # local file path grouping within dataset dir
        samples = []
        for item in video_paths:
            data = item['paths']
            for frame in data:
                frame['rgb'] = os.path.join(self.dataset_dir, frame['rgb'])
                frame['normal'] = os.path.join(self.dataset_dir, frame['normal'])
                frame['albedo'] = os.path.join(self.dataset_dir, frame['albedo'])

                if self.has_brdf:
                    frame['metallic'] = os.path.join(self.dataset_dir, frame['metallic'])
                    frame['roughness'] = os.path.join(self.dataset_dir, frame['roughness'])
                    frame['specular'] = os.path.join(self.dataset_dir, frame['specular'])
            samples.append(data)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _augment_bg(self, img, nml, albedo, threshold = 1e-1, bg_color = [0, 0, 0]):
        # compute bg mask using nml
        # bg color should be the same for all frames inside a video
        mask = np.abs(np.linalg.norm(nml, axis=-1) - 1) > threshold
        img[mask] = bg_color
        albedo[mask] = bg_color
        return img, albedo
    
    def _augment_static_frames(self, frames):
        rand_frame = random.randint(0, self.num_frames - 1) # random select a fixed frame
        return {k: v[rand_frame : rand_frame + 1].expand(self.num_frames, -1, -1, -1)
                for k, v in frames.items() if v is not None}
    
    def _augment_reverse_frames(self, frames):
        return {k: v.flip(0) if v is not None else None for k, v in frames.items()}
    
    def _resize_and_to_tensor(self, img):
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to)
        tensor = transforms.ToTensor()(img)
        return tensor
    
    def _augment_bg_color():
        bg_color = [np.random.uniform(-1.1, 1.1) for _ in range(3)]
        return np.clip(bg_color, -1.0, 1.0)
    
    def __getitem__(self, index):
        sample_fns = self.samples[index]
        total_frames = len(sample_fns) # number of video frames of each data point
        indices = list(range(total_frames))
        
        rgb_list, normal_list, albedo_list = [], [], []
        metallic_list, roughness_list, specular_list = [], [], []
        bg_color = self._augment_bg_color() if self.augment_bg else None
        
        for i in indices:
            frame_paths = sample_fns[i]

            img = np.array(Image.open(frame_paths['rgb']).convert('RGB')) / 255.0 * 2.0 - 1.0
            nml = np.array(Image.open(frame_paths['normal']).convert('RGB')) / 255.0 * 2.0 - 1.0
            albedo = np.array(Image.open(frame_paths['albedo']).convert('RGB')) / 255.0 * 2.0 - 1.0

            if self.augment_bg:
                img, albedo = self._augment_bg(img, nml, albedo, bg_color=bg_color)
            
            rgb_list.append(self._resize_and_to_tensor(img))
            normal_list.append(self._resize_and_to_tensor(nml))
            albedo_list.append(self._resize_and_to_tensor(albedo))
            
            if self.has_brdf:
                metallic = np.array(Image.open(frame_paths['metallic']).convert('L')) / 255.0 * 2 - 1
                roughness = np.array(Image.open(frame_paths['roughness']).convert('L')) / 255.0 * 2 - 1
                specular = np.array(Image.open(frame_paths['specular']).convert('L')) / 255.0 * 2 - 1

                metallic_list.append(self._resize_and_to_tensor(metallic))
                roughness_list.append(self._resize_and_to_tensor(roughness))
                specular_list.append(self._resize_and_to_tensor(specular))
        
        # Stack lists into tensors with shape [F, C, H, W].
        rgb_frames = torch.stack(rgb_list, dim=0)
        normal_frames = torch.stack(normal_list, dim=0)
        albedo_frames = torch.stack(albedo_list, dim=0)
        metallic_frames = torch.stack(metallic_list, dim=0) if self.has_brdf else None
        roughness_frames = torch.stack(roughness_list, dim=0) if self.has_brdf else None
        specular_frames = torch.stack(specular_list, dim=0) if self.has_brdf else None

        frames = {
            'rgb': rgb_frames,
            'normal': normal_frames,
            'albedo': albedo_frames,
            'metallic': metallic_frames,
            'roughness': roughness_frames,
            'specular': specular_frames,
        }        

        # Optionally augment with static frames.
        if self.augment_static and random.random() < self.static_prob:
            frames = self._augment_static_frames(frames)

        # Optionally reverse frame order.
        if self.augment_reverse and random.random() < 0.5:
            frames = self._augment_reverse_frames(frames)

        return {
            'rgbs': frames['rgb'],
            'normals': frames['normal'],
            'albedos': frames['albedo'],
            'metallic': frames['metallic'],
            'specular': frames['specular'],
            'roughness': frames['roughness'],
            'sample_paths': sample_fns,
        }
    

def prepare_dataset_pointers(folder_to_views, possible_frames = [[0, 1, 2], [2, 3, 4]], 
                             save_name = 'dataset_precompute_pairs.json', has_brdf = True):
    samples = []
    
    with open(folder_to_views, "r") as f:
        folder2view = json.load(f)
            
    for folder in folder2view.keys():
        to_load_dir = folder
        nums_to_gen = folder2view[folder] # in the future, some mesh may have diff num of views than other
        
        for i in range(nums_to_gen):
            for frame_group in possible_frames:
                frame_paths = {}
                for fr in frame_group:
                    prefix = f"{i}_motion_{fr}"
                    rgb_path = os.path.join(to_load_dir, f"{prefix}_rgb.png")
                    normal_path = os.path.join(to_load_dir, f"{prefix}_normal.png")
                    albedo_path = os.path.join(to_load_dir, f"{prefix}_albedo.png")

                    metallic_path = os.path.join(to_load_dir, f"{prefix}_metallic.png") if has_brdf else ""
                    specular_path = os.path.join(to_load_dir, f"{prefix}_specular.png") if has_brdf else ""
                    roughness_path = os.path.join(to_load_dir, f"{prefix}_roughness.png") if has_brdf else ""

                    # Collect frame paths
                    frame_paths[fr] = {
                        "rgb": rgb_path,
                        "normal": normal_path,
                        "albedo": albedo_path,
                        "metallic": metallic_path,
                        "specular": specular_path,
                        "roughness": roughness_path,
                    }

                samples.append(
                    {
                        "folder": folder,
                        "frame_group": frame_group,
                        "paths": [frame_paths[fr] for fr in frame_group],
                    }
                )

    with open(save_name, "w") as f:
        json.dump(samples, f, indent=4)
