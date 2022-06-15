import math
import os
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch import distributions as pyd


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def make_dir(*path_parts):
	dir_path = os.path.join(*path_parts)
	try:
		os.mkdir(dir_path)
	except OSError:
		pass
	return dir_path


def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk


def to_np(t):
	if t is None:
		return None
	elif t.nelement() == 0:
		return np.array([])
	else:
		return t.cpu().detach().numpy()


def img_shuffle(img):
	s_ul = img[:, :, :84 // 2, :84 // 2]
	s_ur = img[:, :, :84 // 2, 84 // 2:]
	s_br = img[:, :, 84 // 2:, 84 // 2:]
	s_bl = img[:, :, 84 // 2:, :84 // 2]
	parts = [s_ul, s_ur, s_bl, s_br]

	l = [0, 1, 2, 3]
	random.shuffle(l)
	return torch.cat([
		torch.cat([parts[l[0]], parts[l[1]]], dim=2),
		torch.cat([parts[l[2]], parts[l[3]]], dim=2)
	], dim=3)


class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0,
			high=1,
			shape=((shp[0] * k,) + shp[1:]),
			dtype=env.observation_space.dtype
		)
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs = self.env.reset()
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._frames.append(obs)
		return self._get_obs(), reward, done, info

	def _get_obs(self):
		assert len(self._frames) == self._k
		return np.concatenate(list(self._frames), axis=0)


class ActionQueue:
	def __init__(self, steps):
		self.actions = deque([], maxlen=steps)

	@property
	def length(self):
		return len(self.actions)

	def store(self, action):
		self.actions.append(torch.tensor(action))

	def get(self):
		return torch.cat(list(self.actions), dim=-1).unsqueeze(0)


class LatentQueue:
	def __init__(self):
		self.actions = deque([], maxlen=1)

	@property
	def length(self):
		return len(self.actions)

	def store(self, action):
		self.actions.append(action)

	def get(self):
		return list(self.actions)[0]


class LatentContext:
	def __init__(self, h):
		self.aqs = [ActionQueue(i) for i in [1, 2, 4]]
		self.lqs = [LatentQueue() for _ in range(h)]
		self.h = h
		self.step_counter = 0

	def reset(self):
		self.step_counter = 0
		self.aqs = [ActionQueue(i) for i in [1, 2, 4]]
		self.lqs = [LatentQueue() for _ in range(self.h)]

	def store_action(self, a):
		for i in range(self.h):
			self.aqs[i].store(a)

	def store_latent(self, l):
		pass

	def latent_step(self, ksls):
		if self.step_counter % 4 == 0:
			a = self.aqs[2].get()
			s = self.lqs[2].get()
			self.lqs[2].store(ksls[2].gru(a.to('cuda'), s))

		if self.step_counter % 2 == 0:
			a = self.aqs[1].get()
			s = self.lqs[1].get()
			self.lqs[1].store(ksls[1].gru(a.to('cuda'), s))

		if self.step_counter % 1 == 0:
			a = self.aqs[0].get()
			s = self.lqs[0].get()
			self.lqs[0].store(ksls[0].gru(a.to('cuda'), s))

	def get_latents(self):
		self.step_counter += 1
		return torch.cat([q.get() for q in self.lqs], dim=-1)


class TanhTransform(pyd.transforms.Transform):
	domain = pyd.constraints.real
	codomain = pyd.constraints.interval(-1.0, 1.0)
	bijective = True
	sign = +1

	def __init__(self, cache_size=1):
		super().__init__(cache_size=cache_size)

	@staticmethod
	def atanh(x):
		return 0.5 * (x.log1p() - (-x).log1p())

	def __eq__(self, other):
		return isinstance(other, TanhTransform)

	def _call(self, x):
		return x.tanh()

	def _inverse(self, y):
		# We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
		# one should use `cache_size=1` instead
		return self.atanh(y)

	def log_abs_det_jacobian(self, x, y):
		# We use a formula that is more numerically stable, see details in the following link
		# https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
		return 2.0 * (math.log(2.) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
	def __init__(self, loc, scale):
		self.loc = loc
		self.scale = scale

		self.base_dist = pyd.Normal(loc, scale)
		transforms = [TanhTransform()]
		super().__init__(self.base_dist, transforms)

	@property
	def mean(self):
		mu = self.loc
		for tr in self.transforms:
			mu = tr(mu)
		return mu


def center_crop_image(image, output_size):
	h, w = image.shape[1:]
	new_h, new_w = output_size, output_size

	top = (h - new_h) // 2
	left = (w - new_w) // 2

	image = image[:, top:top + new_h, left:left + new_w]
	return image


def center_crop_images(image, output_size):
	h, w = image.shape[2:]
	new_h, new_w = output_size, output_size

	top = (h - new_h) // 2
	left = (w - new_w) // 2

	image = image[:, :, top:top + new_h, left:left + new_w]
	return image


def center_translate(image, size):
	c, h, w = image.shape
	assert size >= h and size >= w
	outs = np.zeros((c, size, size), dtype=image.dtype)
	h1 = (size - h) // 2
	w1 = (size - w) // 2
	outs[:, h1:h1 + h, w1:w1 + w] = image
	return outs


def create_permuted_traj(T_out, k, bs, latent_dim):
	possible_idxs = list(range(k))[1:]
	switch_idxs = [np.random.choice(possible_idxs) for _ in range(bs)]

	permuted = []

	for i in range(bs):
		inner = []

		for j in range(k):
			inner.append(T_out[j][i].clone())

		inner[switch_idxs[i]], inner[switch_idxs[i] - 1] = inner[switch_idxs[i] - 1], inner[switch_idxs[i]]

		permuted.extend(inner)

	permuted = torch.stack(permuted).reshape(bs, k * latent_dim)

	return permuted


def lip2d(x, logit, kernel=3, stride=2, padding=1):
	weight = logit.exp()
	return F.avg_pool2d(x * weight, kernel, stride, padding) / F.avg_pool2d(weight, kernel, stride, padding)


#from sklearn.linear_model import LinearRegression


def predict_past(present, past):
	lr = LinearRegression()
	lr.fit(present, past)
	past_hat = lr.predict(present)
	return F.mse_loss(torch.tensor(past_hat), past).item()


# def weight_init(m):
# 	pass
# """Custom weight init for Conv2D and Linear layers."""
# if isinstance(m, nn.Linear):
#     nn.init.orthogonal_(m.weight.data)
#     if m.bias is not None:
#         m.bias.data.fill_(0.0)
# elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#     # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
#     assert m.weight.size(2) == m.weight.size(3)
#     m.weight.data.fill_(0.0)
#     if m.bias is not None:
#         m.bias.data.fill_(0.0)
#     mid = m.weight.size(2) // 2
#     gain = nn.init.calculate_gain('relu')
#     nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


def rotate_sequence(sequence):
	degrees = [90, 180, 270]
	d = int(np.random.choice(degrees))

	rotated = [
		T.functional.rotate(sequence[:, i, :, :, :], d).unsqueeze(1)
		for i in range(sequence.shape[1])
	]

	return torch.cat(rotated, dim=1)


class ObservationSpace:
	def __init__(self, env):
		self.env = env
		self.shape = (3, 84, 84)
		self.dtype = np.uint8


class ActionSpace:
	def __init__(self, min, max, shape):
		self.min = min
		self.max = max
		self.shape = shape
		self.low = np.array([min])
		self.high = np.array([max])

	def sample(self):
		return np.random.uniform(self.min, self.max, self.shape)


class DCSEnvironment:
	def __init__(self, env, max_episode_steps, seed, camera_id, repeat):
		self.env = env
		self._max_episode_steps = None
		self.action_space = ActionSpace(
			env.action_spec().minimum[0],
			env.action_spec().maximum[0],
			env.action_spec().shape
		)
		self.seed = seed
		self.camera_id = camera_id
		self.repeat = repeat

		self.observation_space = ObservationSpace(env)
		self._max_episode_steps = max_episode_steps
		self.reward_range = (-np.inf, np.inf)
		self.metadata = {'render.modes': []}
		self.curr_step = 0

	def step(self, action):
		self.curr_step += 1

		reward = 0
		for _ in range(self.repeat):
			out = self.env.step(action)
			reward += out.reward

		obs = np.rollaxis(out.observation['pixels'], 2, 0)
		done = self.curr_step >= self._max_episode_steps
		info = {}
		return obs, reward, done, info

	def reset(self):
		self.curr_step = 0
		obs = self.env.reset()
		return np.rollaxis(obs.observation['pixels'], 2, 0)

	def render(self, mode, height, width):
		return self.env.physics.render(height=height, width=width, camera_id=self.camera_id)
