import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math


class Encoder(nn.Module):
	"""Convolutional encoder. Dropout operations are only here for experimentation purposes. Keep DROPOUT and
	DROPOUT_FC = 0.0 for replicating results.

		Attributes:
			num_layers (int): number of convolutional layers in the encoder
			num_filters (int): number of convolutional kernels per convolutional layer
			output_logits (bool): whether or not to run the output of the encoder through a tanh activation
			feature_dim (int): the dimensionality of the latent vector
	"""

	def __init__(self, obs_shape, feature_dim):
		super().__init__()

		assert len(obs_shape) == 3
		self.num_layers = 4
		self.num_filters = 32
		self.output_logits = True
		self.feature_dim = feature_dim

		self.convs = nn.ModuleList([
			nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
			nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
			nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
			nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
		])

		self.head = nn.Sequential(
			nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
			nn.LayerNorm(self.feature_dim))

		self.outputs = dict()

	def forward_conv_no_flatten(self, obs):
		conv = obs / 255.

		for layer in self.convs:
			if 'stride' not in layer.__constants__:
				conv = layer(conv)
			else:
				conv = torch.relu(layer(conv))
		return conv

	def forward_conv(self, obs):
		"""Forward pass through only the convolutional layers of the network

		Args:
			obs (torch.Tensor): non-normed image input

		Returns:
			output of the convolutional layers of the encoder
		"""
		conv = obs / 255.

		for layer in self.convs:
			if 'stride' not in layer.__constants__:
				conv = layer(conv)
			else:
				conv = torch.relu(layer(conv))

		h = conv.view(conv.size(0), -1)
		return h

	def collect_convs(self, x):
		outs = []

		for layer in self.convs:
			x = torch.relu(layer(x))
			outs.append(x)

		return outs

	def forward(self, obs, detach=False):
		"""Forward pass through the entire encoder

		Args:
			obs (torch.Tensor): non-normed image input
			detach (bool): whether or not to detach the convolutional layers from the computation graph

		Returns:
			latent representation of the input image(s)
		"""
		h = self.forward_conv(obs)

		if detach:
			h = h.detach()

		out = self.head(h)

		if not self.output_logits:
			out = torch.tanh(out)

		self.outputs['out'] = out

		return out

	def copy_conv_weights_from(self, source):
		"""Tie the convolutional weights between this model and a target model

		Args:
			source (torch.nn.Module): a model with congruent convolutional layers to this model

		Returns:
			None
		"""
		for i in range(len(self.convs)):
			if 'stride' not in self.convs[i].__constants__:
				pass
			else:
				utils.tie_weights(src=source.convs[i], trg=self.convs[i])

	def log(self, logger, step):
		"""Logs information for the CLI

		Args:
			logger (logger.Logger): Logger class
			step (int): the current step

		Returns:
			None
		"""
		for k, v in self.outputs.items():
			logger.log_histogram(f'train_encoder/{k}_hist', v, step)
			if len(v.shape) > 2:
				logger.log_image(f'train_encodcer/{k}_img', v[0], step)

		for i in range(self.num_layers):
			logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
	"""torch.distributions implementation of an diagonal Gaussian policy

		Attributes:
			encoder_cfg (hydra.config): hydra config as specified by config.yaml
			action_shape (tuple): action shape of the env, e.g., (6,)
			hidden_dim (int): number of hidden units per layer in the MLP
			hidden_depth (int): number of hidden layers in the MLP
	"""

	def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
				 log_std_bounds, n_latents):

		super().__init__()

		self.encoder = hydra.utils.instantiate(encoder_cfg)

		self.log_std_bounds = log_std_bounds
		self.trunk = utils.mlp(self.encoder.feature_dim * n_latents, hidden_dim,
							   2 * action_shape[0], hidden_depth)

		self.outputs = dict()

	def forward(self, obs, detach_encoder=False):
		"""Forward pass through the entire Actor (encoder + MLP)

		Args:
			obs (torch.Tensor): non-normed image input
			detach_encoder (bool): whether or not to detach the convolutional layers from the compute graph

		Returns:
			SquashedNormal distribution
		"""
		obs = self.encoder(obs, detach=detach_encoder)

		mu, log_std = self.trunk(obs).chunk(2, dim=-1)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std_min, log_std_max = self.log_std_bounds
		log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
		std = log_std.exp()

		self.outputs['mu'] = mu
		self.outputs['std'] = std
		dist = utils.SquashedNormal(mu, std)

		return dist

	def forward_fc(self, obs):
		mu, log_std = self.trunk(obs).chunk(2, dim=-1)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std_min, log_std_max = self.log_std_bounds
		log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
		std = log_std.exp()

		self.outputs['mu'] = mu
		self.outputs['std'] = std

		dist = utils.SquashedNormal(mu, std)

		return dist

	def noise(self, obs, n, detach_encoder=False):
		"""Same as self.forward() but with a small amount of noise added in the form of _n_ 0s to the latent vector
		output of the Actor's encoder

		Args:
			obs (torch.Tensor): non-normed image input
			n (int): the number of elements to 0 out
			detach_encoder (bool): whether or not to detach the convolutional layers from the compute graph

		Returns:
			SquashedNormal distribution
		"""
		obs = self.encoder(obs, detach=detach_encoder)

		obs[0][np.random.choice(range(len(obs[0])), n, replace=False)] = 0

		mu, log_std = self.trunk(obs).chunk(2, dim=-1)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std_min, log_std_max = self.log_std_bounds
		log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
		std = log_std.exp()

		self.outputs['mu'] = mu
		self.outputs['std'] = std

		dist = utils.SquashedNormal(mu, std)
		return dist

	def log(self, logger, step):
		"""Logs information for the CLI

			Args:
				logger (logger.Logger): Logger class
				step (int): the current step

			Returns:
				None
		"""
		for k, v in self.outputs.items():
			logger.log_histogram(f'train_actor/{k}_hist', v, step)

		for i, m in enumerate(self.trunk):
			if type(m) == nn.Linear:
				logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
	"""Critic network, employs double Q-learning.

		Attributes:
				encoder_cfg (hydra.config): hydra config as specified by config.yaml
				action_shape (tuple): action shape of the env, e.g., (6,)
				hidden_dim (int): number of hidden units per layer in the MLP
				hidden_depth (int): number of hidden layers in the MLP
	"""

	def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
		super().__init__()

		self.encoder = hydra.utils.instantiate(encoder_cfg)

		self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
							hidden_dim, 1, hidden_depth)
		self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
							hidden_dim, 1, hidden_depth)

		self.outputs = dict()

	def forward(self, obs, action, detach_encoder=False):
		"""

		Args:
			obs (torch.Tensor): non-normed image input
			action (torch.Tensor):  action vector taken by agent
			detach_encoder (bool): whether or not to detach the convolutional layers from the compute graph

		Returns:

		"""
		assert obs.size(0) == action.size(0)
		obs = self.encoder(obs, detach=detach_encoder)

		obs_action = torch.cat([obs, action], dim=-1)
		q1 = self.Q1(obs_action)
		q2 = self.Q2(obs_action)

		self.outputs['q1'] = q1
		self.outputs['q2'] = q2

		return q1, q2

	def forward_fc(self, obs, action):
		obs_action = torch.cat([obs, action], dim=-1)
		q1 = self.Q1(obs_action)
		q2 = self.Q2(obs_action)

		self.outputs['q1'] = q1
		self.outputs['q2'] = q2

		return q1, q2


	def log(self, logger, step):
		"""Logs information for the CLI

			Args:
				logger (logger.Logger): Logger class
				step (int): the current step

			Returns:
				None
		"""
		self.encoder.log(logger, step)

		for k, v in self.outputs.items():
			logger.log_histogram(f'train_critic/{k}_hist', v, step)

		assert len(self.Q1) == len(self.Q2)
		for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
			assert type(m1) == type(m2)
			if type(m1) is nn.Linear:
				logger.log_param(f'train_critic/q1_fc{i}', m1, step)
				logger.log_param(f'train_critic/q2_fc{i}', m2, step)



class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)



def init_normalization(channels, type="bn", affine=True, one_d=False):
    assert type in ["bn", "ln", "in", "none", None]
    if type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=affine)
        else:
            return nn.BatchNorm2d(channels, affine=affine)
    elif type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return nn.GroupNorm(1, channels, affine=affine)
    elif type == "in":
        return nn.GroupNorm(channels, channels, affine=affine)
    elif type == "none" or type is None:
        return nn.Identity()


def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type="bn"):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            init_normalization(out_channels, norm_type),
            Conv2dSame(out_channels, out_channels, 3),
            init_normalization(out_channels, norm_type),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = F.relu(out)
        return out


class SPR(nn.Module):
	def __init__(
			self,
			hidden_size,
			action_shape,
			blocks,
			device,
			image_shape,
			output_size,
			n_atoms,
			dueling,
			jumps,
			spr,
			augmentation,
			target_augmentation,
			eval_augmentation,
			dynamics_blocks,
			norm_type,
			noisy_nets,
			aug_prob,
			classifier,
			imagesize,
			time_offset,
			local_spr,
			global_spr,
			momentum_encoder,
			shared_encoder,
			distributional,
			dqn_hidden_size,
			momentum_tau,
			renormalize,
			q_l1_type,
			dropout,
			final_classifier,
			model_rl,
			noisy_nets_std,
			residual_tm,
			use_maxpool=False,
			channels=None,  # None uses default.
			kernel_sizes=None,
			strides=None,
			paddings=None,
			framestack=4,
	):
		super().__init__()
		self.noisy = noisy_nets
		self.time_offset = time_offset
		self.aug_prob = aug_prob
		self.classifier_type = classifier

		self.distributional = distributional
		n_atoms = 1 if not self.distributional else n_atoms
		self.dqn_hidden_size = dqn_hidden_size
		self.renormalize = renormalize

		self.action_shape = action_shape[0]

		self.transforms = []
		self.eval_transforms = []

		self.uses_augmentation = False
		# for aug in augmentation:
		# 	if aug == "affine":
		# 		transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
		# 		eval_transformation = nn.Identity()
		# 		self.uses_augmentation = True
		# 	elif aug == "crop":
		# 		transformation = RandomCrop((84, 84))
		# 		# Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
		# 		eval_transformation = CenterCrop((84, 84))
		# 		self.uses_augmentation = True
		# 		imagesize = 84
		# 	elif aug == "rrc":
		# 		transformation = RandomResizedCrop((100, 100), (0.8, 1))
		# 		eval_transformation = nn.Identity()
		# 		self.uses_augmentation = True
		# 	elif aug == "blur":
		# 		transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
		# 		eval_transformation = nn.Identity()
		# 		self.uses_augmentation = True
		# 	elif aug == "shift":
		# 		transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
		# 		eval_transformation = nn.Identity()
		# 	elif aug == "intensity":
		# 		transformation = Intensity(scale=0.05)
		# 		eval_transformation = nn.Identity()
		# 	elif aug == "none":
		# 		transformation = eval_transformation = nn.Identity()
		# 	else:
		# 		raise NotImplementedError()
		# 	self.transforms.append(transformation)
		# 	self.eval_transforms.append(eval_transformation)

		# TODO: NEEDS TO BE THE ACTOR/CRITIC ENCODER

		# TRANSITION MODEL
		layers = [Conv2dSame(channels + self.action_shape, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]

		for _ in range(blocks):
			layers.append(ResidualBlock(hidden_size, hidden_size, norm_type))

		layers.extend([Conv2dSame(hidden_size, channels, 3)])

		self.transition_model = nn.Sequential(*layers).to(device)

		self.transition_model_opt = torch.optim.Adam(self.transition_model.parameters(), lr=1e-3)

		# TODO: nonlinear cnn thing? w for HKSL
		self.nonlinear = nn.Sequential(nn.Linear(32 * 35 * 35, 50), nn.ReLU(), nn.Linear(50, 50)).to(device)
		self.nonlinear_target = nn.Sequential(nn.Linear(32 * 35 * 35, 50), nn.ReLU(), nn.Linear(50, 50)).to(device)

		self.nonlinear_opt = torch.optim.Adam(self.nonlinear.parameters(), lr=1e-3)

		self.train()

	def forward_model(self, x, actions):
		# print(f'FORWARD X: {x.shape}')
		action_tensor = torch.zeros(actions.shape[0], self.action_shape, x.shape[-2], x.shape[-1],
									device=actions.device)

		# print(f'FORWARD action_tensor: {action_tensor.shape}')
		for i in range(actions.shape[0]):
			for j in range(actions.shape[1]):
				action_tensor[i, j] += actions[i, j]

		stacked_image = torch.cat([x, action_tensor], dim=1)
		next_state = F.relu(self.transition_model(stacked_image))

		if self.renormalize:
			next_state = renormalize(next_state, 1)

		return next_state

	def spr_loss(self, f_x1s, f_x2s):
		f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
		f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
		# Gradients of norrmalized L2 loss and cosine similiarity are proportional.
		# See: https://stats.stackexchange.com/a/146279
		loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
		return loss

	def train_rollout(self, conv, target_conv, observations, actions, encoder_opt):
		# Obs shape: [128, 4, 9, 84, 84] [B, T, c, h, w]
		# A shape: [128, T, action_shape]

		# (1) Pass first observations in traj through the image encoder shared by Critic/Actor
		# [128, 32, 35, 35]
		conv_activations = conv.forward_conv_no_flatten(observations[:, 0])

		# (2) Roll forward with the actions, make preds, then compute loss
		loss = 0
		for i in range(observations.shape[1] - 1):
			conv_activations = self.forward_model(conv_activations, actions[:, i])
			pred = self.nonlinear(conv_activations.view(conv_activations.shape[0], -1))

			with torch.no_grad():
				target = self.nonlinear_target(
					target_conv.forward_conv_no_flatten(observations[:, i + 1]).view(conv_activations.shape[0], -1)
				)

			inner_loss = self.spr_loss(pred, target)
			loss += inner_loss

		self.transition_model_opt.zero_grad()
		self.nonlinear_opt.zero_grad()
		encoder_opt.zero_grad()
		loss.backward()
		self.transition_model_opt.step()
		self.nonlinear_opt.step()
		encoder_opt.step()


class SPRAgent:
	def __init__(self, action_shape, action_range, device, critic_cfg, actor_cfg, discount, init_temperature, lr,
				 actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size, ksl_update_frequency,
				 k, obs_shape, encoder_cfg, h, connected, critic_seq, mi_min,
				 critic_nstep, shared_enc, recon, covar, r_pred, clip_grad, mut, repr, residual, a_pred,
				 action_repeat, env):
		self.name = 'SPR-Agent'
		self.action_range = action_range
		self.device = device
		self.discount = discount
		self.critic_tau = critic_tau
		self.actor_update_frequency = actor_update_frequency
		self.critic_target_update_frequency = critic_target_update_frequency
		self.batch_size = batch_size
		self.action_shape = action_shape[0]
		self.k = k

		self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
		self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
		self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

		self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
		self.log_alpha.requires_grad = True

		# set target entropy to -|A|
		self.target_entropy = -action_shape[0]

		# SPR STUFF HERER

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

		self.spr_bs = SPR(
			hidden_size=256,
			action_shape=action_shape,
			blocks=0,
			image_shape=(3, 84, 84),
			device=device,
			output_size=None,
			n_atoms=None,
			dueling=None,
			jumps=None,
			spr=None,
			augmentation=None,
			target_augmentation=None,
			eval_augmentation=None,
			dynamics_blocks=None,
			norm_type=None,
			noisy_nets=None,
			aug_prob=None,
			classifier=None,
			imagesize=None,
			time_offset=None,
			local_spr=None,
			global_spr=None,
			momentum_encoder=None,
			shared_encoder=None,
			distributional=None,
			dqn_hidden_size=None,
			momentum_tau=None,
			renormalize=True,
			q_l1_type=None,
			dropout=None,
			final_classifier=None,
			model_rl=None,
			noisy_nets_std=None,
			residual_tm=None,
			use_maxpool=False,
			channels=32,  # None uses default.
			kernel_sizes=None,
			strides=None,
			paddings=None,
			framestack=4,)

		self.train()

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)
		self.critic_target.train(training)

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def act(self, obs, sample=False):
		obs = torch.FloatTensor(obs).to(self.device)
		obs = obs.unsqueeze(0)
		dist = self.actor(obs)
		action = dist.sample() if sample else dist.mean
		action = action.clamp(*self.action_range)
		assert action.ndim == 2 and action.shape[0] == 1
		return utils.to_np(action[0])

	def update_critic(self, obs, action, reward, next_obs, not_done):
		with torch.no_grad():
			dist = self.actor(next_obs)
			next_action = dist.rsample()
			log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_prob
			target_Q = reward + (not_done * self.discount * target_V)

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
			current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_actor_and_alpha(self, obs):
		dist = self.actor(obs, detach_encoder=True)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)
		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = (self.alpha *
					  (-log_prob - self.target_entropy).detach()).mean()

		alpha_loss.backward()
		self.log_alpha_optimizer.step()

	def update(self, replay_buffer, step):
		obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
			self.batch_size)

		# logger.log('train/batch_reward', reward.mean(), step)

		self.update_critic(obs, action, reward, next_obs, not_done)

		if step % self.actor_update_frequency == 0:
			self.update_actor_and_alpha(obs)

		# Obs shape: [128, 4, 9, 84, 84] [B, T, c, h, w]
		# A shape: [128, T, action_shape]
		obses, actions, obses_next, rewards, not_dones = replay_buffer.sample_traj_efficient(self.batch_size, self.k)

		self.spr_bs.train_rollout(self.critic.encoder, self.critic_target.encoder, obses, actions, self.critic_optimizer)

		if step % self.critic_target_update_frequency == 0:
			utils.soft_update_params(self.critic, self.critic_target,
									 self.critic_tau)
			utils.soft_update_params(self.spr_bs.nonlinear, self.spr_bs.nonlinear_target,
									 self.critic_tau)

