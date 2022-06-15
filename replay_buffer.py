import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, image_pad, device, env):
        self.capacity = capacity
        self.device = device
        self.env = env
        self.image_pad = image_pad

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1]))
        )

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.eoo = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, eoo):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.eoo[self.idx], eoo)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        # KEEP UNCOMMENTED FOR TRANSLATION
        # For ablations with no augmentation, *comment out* the following lines
        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug

    def sample_traj(self, batch_size, k):
        """Really will only work for envs with fixed length episodes, such as in dm_control"""
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        end_ranges = end_idxs - k

        traj_idxs = []

        n_slots = len(end_idxs)

        for _ in range(batch_size):
            slot_idx = np.random.choice(n_slots)
            beg = np.random.choice(range(beg_ranges[slot_idx], end_ranges[slot_idx]))
            traj_idxs.append([beg + i for i in range(k)])

        actions = np.array([self.actions[traj_idxs[i]] for i in range(batch_size)])

        # KEEP UNCOMMENTED FOR TRANSLATION
        obses = np.array(
            [
                random_crop(self.obses[traj_idxs[i]], self.image_pad) for i in range(batch_size)
            ]
        )

        # obses_next = np.array(
        #     [
        #         random_crop(self.next_obses[traj_idxs[i]], self.image_pad) for i in range(batch_size)
        #     ]
        # )

        # ORIGINAL
        # obses = np.array(
        #     [
        #         random_crop(self.obses[traj_idxs[i]], self.image_pad) for i in range(batch_size)
        #     ]
        # )
        # obses_next = np.array(
        #     [
        #         random_crop(self.next_obses[traj_idxs[i]], self.image_pad) for i in range(batch_size)
        #     ]
        # )

        # For ablations with no augmentation, *uncomment* the following lines
        # obses = np.array([
        #     self.obses[traj_idxs[i]] for i in range(batch_size)
        # ])
        # obses_next = np.array([
        #     self.next_obses[traj_idxs[i]] for i in range(batch_size)
        # ])

        rewards = np.array([self.rewards[traj_idxs[i]] for i in range(batch_size)])
        not_dones = np.array([self.not_dones_no_max[traj_idxs[i]] for i in range(batch_size)])

        actions = torch.as_tensor(actions, device=self.device)
        obses = torch.tensor(obses, device=self.device).float()
        obses_next = torch.tensor(obses_next, device=self.device).float()
        rewards = torch.tensor(rewards, device=self.device)
        not_dones = torch.tensor(not_dones, device=self.device)

        return obses, actions, obses_next, rewards#, not_dones

    def sample_traj_efficient(self, batch_size, k):
        """Really will only work for envs with fixed length episodes, such as in dm_control"""
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        end_ranges = end_idxs - k - 1

        traj_idxs = []

        n_slots = len(end_idxs)

        for _ in range(batch_size):
            slot_idx = np.random.choice(n_slots)
            beg = np.random.choice(range(beg_ranges[slot_idx], end_ranges[slot_idx]))
            traj_idxs.append([beg + i for i in range(k + 1)])


        actions = np.array([self.actions[traj_idxs[i]] for i in range(batch_size)])

        # KEEP UNCOMMENTED FOR TRANSLATION
        obses = np.array(
            [
                random_crop(self.obses[traj_idxs[i]], self.image_pad) for i in range(batch_size)
            ]
        )

        rewards = np.array([self.rewards[traj_idxs[i]] for i in range(batch_size)])
        not_dones = np.array([self.not_dones_no_max[traj_idxs[i]] for i in range(batch_size)])

        actions = torch.as_tensor(actions, device=self.device)
        obses = torch.tensor(obses, device=self.device).float()
        rewards = torch.tensor(rewards, device=self.device)
        not_dones = torch.tensor(not_dones, device=self.device)

        return obses, actions, None, rewards, not_dones

    def sample_episode(self, batch_size):
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        episode_len = end_idxs[0] - beg_ranges[0]

        n_slots = len(end_idxs)

        traj_idxs = []

        slot_idx = np.random.choice(n_slots, size=batch_size, replace=False)
        for s in slot_idx:
            beg = beg_ranges[s]
            traj_idxs.append([beg + i for i in range(episode_len + 1)])

        actions = np.array([self.actions[traj_idxs[i]] for i in range(batch_size)])

        # KEEP UNCOMMENTED FOR TRANSLATION
        obses = np.array(
            [
                random_crop(self.obses[traj_idxs[i]], self.image_pad) for i in range(batch_size)
            ]
        )

        rewards = np.array([self.rewards[traj_idxs[i]] for i in range(batch_size)])
        not_dones = np.array([self.not_dones[traj_idxs[i]] for i in range(batch_size)])

        actions = torch.as_tensor(actions, device=self.device)
        obses = torch.tensor(obses, device=self.device).float()
        rewards = torch.tensor(rewards, device=self.device)
        not_dones = torch.tensor(not_dones, device=self.device)

        return obses, actions, None, rewards, not_dones

    def sample_traj_ends(self, batch_size, k):
        """Really will only work for envs with fixed length episodes, such as in dm_control"""
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        end_ranges = end_idxs - k

        traj_idxs = []

        n_slots = len(end_idxs)

        for _ in range(batch_size):
            slot_idx = np.random.choice(n_slots)
            beg = np.random.choice(range(beg_ranges[slot_idx], end_ranges[slot_idx]))
            traj_idxs.append([beg + i for i in range(k)])


        # Ends are for the states, you need the actions and rewards
        ends = []
        for traj in traj_idxs:
            ends.append([traj[0], traj[-1]])


        actions = np.array([self.actions[traj_idxs[i]] for i in range(batch_size)])

        # KEEP UNCOMMENTED FOR TRANSLATION
        obses = np.array(
            [
                random_crop(self.obses[ends[i]], self.image_pad) for i in range(batch_size)
            ]
        )
        obses_next = np.array(
            [
                random_crop(self.next_obses[ends[i]], self.image_pad) for i in range(batch_size)
            ]
        )

        rewards = np.array([self.rewards[traj_idxs[i]] for i in range(batch_size)])

        actions = torch.as_tensor(actions, device=self.device)
        obses = torch.tensor(obses, device=self.device).float()
        obses_next = torch.tensor(obses_next, device=self.device).float()
        rewards = torch.tensor(rewards, device=self.device)

        return obses, actions, obses_next, rewards


def random_crop(imgs, image_pad, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h + (image_pad * 2) - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        img = np.pad(img, image_pad, mode='constant')[image_pad:-image_pad, :, :]
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs
