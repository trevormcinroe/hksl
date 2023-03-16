import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math

DROPOUT = 0.0
DROPOUT_FC = 0.0


def loss_fn(x, y):
	x = F.normalize(x, dim=-1, p=2)
	y = F.normalize(y, dim=-1, p=2)
	return 2 - 2 * (x * y).sum(dim=-1)


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


class GRUCellVanilla(nn.Module):

	def __init__(self, latent_shape, action_shape):
		super().__init__()
		#
		self.reset_gate = nn.Linear(latent_shape + action_shape, latent_shape)
		self.update_gate = nn.Linear(latent_shape + action_shape, latent_shape)
		self.candidate_mm = nn.Linear(latent_shape + action_shape, latent_shape)

		self.reset_connected = nn.Linear(latent_shape * 2, latent_shape)
		self.update_connected = nn.Linear(latent_shape * 2, latent_shape)
		self.candidate_connected_mm = nn.Linear(latent_shape * 2, latent_shape)


	def forward(self, action, prev_z, topdown=None):
		reset = torch.sigmoid(
			self.reset_gate(torch.cat([prev_z, action], dim=1))
		)

		update = torch.sigmoid(
			self.update_gate(torch.cat([prev_z, action], dim=1))
		)

		# Generating candidate vector
		candidate = torch.tanh(
			self.candidate_mm(torch.cat([reset * prev_z, action], dim=1))
		)

		h_out = update * prev_z + (1 - update) * candidate

		if topdown is not None:
			reset_above = torch.sigmoid(
				self.reset_connected(torch.cat([topdown, prev_z], dim=1))
			)

			update_above = torch.sigmoid(
				self.update_connected(torch.cat([topdown, prev_z], dim=1))
			)

			candidate_above = torch.tanh(
				self.candidate_connected_mm(torch.cat([reset_above * topdown, prev_z], dim=1))
			)

			h_out_above = update_above * topdown + (1 - update_above) * candidate_above

			h_out = (h_out + h_out_above) / 2

		return h_out


class KSL(nn.Module):
	"""KSL Module

	Attributes:
		critic_online (torch.nn.Module): Critic class - used as critic in agent
		critic_momentum (torch.nn.Module): Critic class - used as target critic in agent
		action_shape (tuple): action shape of the env, e.g., (6,)

	"""
	def __init__(self, critic_online, critic_momentum, action_shape, latent_dim, k, skips, multiple, levels, h):
		super().__init__()
		self.encoder_online = critic_online.encoder
		self.encoder_momentum = critic_momentum.encoder

		self.Wz = nn.Sequential(nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 50))
		self.r_pred = nn.Sequential(nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, k))
		self.gru = GRUCellVanilla(latent_dim, action_shape * skips)

		if h == len(levels.keys()) - 1:
			self.pre_gru = None
		else:
			self.pre_gru = nn.Sequential(
				nn.Linear(latent_dim * len(levels[h+1]) + len(levels[h]) - 1, 128),
				nn.ReLU(),
				nn.Linear(128, 50),
			)

	def encode(self, s, s_):
		"""Used to encode a current state (s) and next-step state (s_) along the online and momentum pathways,
		respectively

		Args:
			s (torch.Tensor): non-normed image input
			s_ (torch.Tensor): non-normed  inage input

		Returns:
			latent vectors from the online and momentum encoders, respectively
		"""
		h = self.encoder_online(s)
		h_ = self.encoder_momentum(s_).detach()

		return h, h_

	def transition(self, h, a):
		"""Forward pass through the KSL module's transition module \mathcal{T}

		Args:
			h (torch.Tensor): latent vector
			a (torch.Tensor): action vector

		Returns:
			predicted next-step latent vector
		"""
		h = self.transition_model(h, a)
		return h

	def projection(self, h, h_):
		"""Forward pass through the KSL  module's projection modules \Psi

		Args:
			h (torch.Tensor): latent vector
			h_ (torch.Tensor): latent vector

		Returns:
			projection vector
		"""
		projection = self.proj_online(h)
		projection_ = self.proj_momentum(h_).detach()

		return projection, projection_

	def predict(self, projection):
		"""Forward pass through the KSL module's prediction head \mathcal{P}

		Args:
			projection (torch.Tensor): projection vector

		Returns:
			prediction vector
		"""
		z_hat = self.Wz(projection)

		return z_hat


class KSLAgent:
	"""k-Step Latent Agent

		Attributes:
			action_shape (tuple): action shape of the env, e.g., (6,)
			action_range (tuple): provided by the env
			device (str): describes the hardware on which the training occurs. e.g., cuda, gpu, cpu
			critic_cfg (hydra.config): as specified in config.yaml
			actor_cfg (hydra.config): as specified in config.yaml
			discount (float): discount rate, gamma
			init_temperature (float): the initial value for alpha, the entropy parameter of SAC
			lr (float): the learning rate
			actor_update_frequency (int): the number of steps between updating the actor networks
			critic_tau (float): value used for the EMA update for the critic target
			critic_target_update_frequency (int): the number of steps between the EMA update for the target critic
			batch_size (int): the mini-batch size used for training
			ksl_update_frequency (int): the number of steps between updating via KSL
			k (int): the value of _k_ for KSL
	"""

	def __init__(self, action_shape, action_range, device, critic_cfg, actor_cfg, discount, init_temperature, lr,
				 actor_update_frequency, critic_tau, critic_target_update_frequency, batch_size, ksl_update_frequency,
				 k, obs_shape, encoder_cfg, h, connected, critic_seq, mi_min,
				 critic_nstep, shared_enc, recon, covar, r_pred, clip_grad, mut, repr, residual, a_pred,
				 action_repeat, env):
		self.name = 'KSL-Agent'
		self.action_range = action_range
		self.device = device
		self.discount = discount
		self.critic_tau = critic_tau
		self.actor_update_frequency = actor_update_frequency
		self.critic_target_update_frequency = critic_target_update_frequency
		self.batch_size = batch_size
		self.ksl_update_frequency = ksl_update_frequency
		self.k = k
		self.action_shape = action_shape[0]
		self.h = h
		self.connected = connected
		self.critic_seq = critic_seq
		self.critic_nstep = critic_nstep
		self.shared_enc = shared_enc
		self.recon = recon
		self.covar = covar
		self.r_pred = r_pred
		self.clip_grad = clip_grad
		self.mut = mut
		self.repr = repr
		self.residual = residual
		self.a_pred = a_pred
		self.mi_min = mi_min
		self.action_repeat = action_repeat

		if 'cartpole' in env or 'ball' in env or 'walker' in env or 'finger' or 'reacher' in env:
			skips = [0, 2, 5, 11]
		elif 'cheetah' in env:
			skips = [3, 4]

		self.skips = skips

		self.levels = {}
		for j in range(self.h):
			self.levels[j] = [i + skips[j]*i for i in range(self.k) if i + skips[j]*i < self.k]

		print(self.levels)

		possible = []
		for j in range(self.h):
			possible.extend(self.levels[j])

		self.replay_len = np.max(possible) + 1

		self.critic_losses = {i+1: [] for i in range(h)}

		if not self.shared_enc:
			actor_cfg.params.n_latents = self.h

		self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
		if 'finger' in env or 'cheetah' in env or 'reacher' in env:
			self.actor.apply(utils.weight_init)
		self.critics = [
			hydra.utils.instantiate(critic_cfg).to(self.device) for _ in range(h)
		]

		# Encoder (online), Q1/Q2
		if 'finger' in env or 'cheetah' in env or 'reacher' in env:
			for i in range(h):
				self.critics[i].apply(utils.weight_init)

		self.critic_targets = [
			hydra.utils.instantiate(critic_cfg).to(self.device) for _ in range(h)
		]

		# Encoder (momentum) Q1/Q2 (momentum)
		for i in range(h):
			self.critic_targets[i].load_state_dict(self.critics[i].state_dict())

		# tie conv layers between actor and critic
		# self.actor.encoder.copy_conv_weights_from(self.critic.encoder)
		self.actor.encoder.copy_conv_weights_from(self.critics[0].encoder)

		# Sharing all conv layers with all critics
		if self.shared_enc:
			for i in range(1, h):
				self.critics[i].encoder.copy_conv_weights_from(self.critics[0].encoder)

		self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
		self.log_alpha.requires_grad = True

		# set target entropy to -|A|
		self.target_entropy = -action_shape[0]

		multiples = [1, 2, 4, 8]

		# To avoid overwriting the encoders, the weight_init procedure is handled within the KSL class
		self.ksls = [
			KSL(self.critics[i], self.critic_targets[i], action_shape[0], 50, self.k, skips[i]+1,
				multiples[i], self.levels, i).to(self.device)
			for i in range(h)
		]

		# optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic_optimizers = [
			torch.optim.Adam(self.critics[i].parameters(), lr=lr)
			for i in range(h)
		]

		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

		self.ksl_optimizers = [
			torch.optim.Adam(self.ksls[i].parameters(), lr=lr)
			for i in range(h)
		]
		self.encoder_optimizers = [
			torch.optim.Adam(self.critics[i].encoder.parameters(), lr=lr)
			for i in range(h)
		]

		self.train()

		self.loss_fn = loss_fn
		self.bce_loss = torch.nn.BCEWithLogitsLoss()

		self.ksl_loss_hist = []
		self.bce_loss_hist = []
		self.r_label_hist = []
		self.r_hat_hist = []
		self.r_loss_hist = []
		self.alpha_hist = []
		self.recon_loss_hist = []
		self.cov_loss_hist = []
		self.r_pred_loss_hist = []
		self.final_cont_loss_hist = []
		self.ce_loss_hist = []
		self.eoo_hist = []

		self.critic_grads = {i+1: [] for i in range(h)}
		self.a_grads = []
		self.a_mag = []
		self.a_std = []

		self.ksl_loss_hist = {i+1: [] for i in range(h)}
		self.ksl_grads = {i + 1: [] for i in range(h)}

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		for i in range(self.h):
			self.critics[i].train(training)
			self.critic_targets[i].train(training)
			self.ksls[i].train(training)

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def act(self, obs, sample=False):
		"""Samples an action from the Actor network

		Args:
			obs (torch.Tensor): non-normed image input
			sample (bool): True = true sampling, False = deterministic sampling

		Returns:
			np.array version of action vector
		"""
		obs = torch.FloatTensor(obs).to(self.device)
		obs = obs.unsqueeze(0)

		if self.shared_enc:

			dist = self.actor(obs)

		else:

			obs = torch.cat([
				self.critics[i].encoder(obs) for i in range(self.h)
			], dim=1)

			dist = self.actor.forward_fc(obs)

		# dist = self.actor(obs)
		action = dist.sample() if sample else dist.mean
		action = action.clamp(*self.action_range)
		assert action.ndim == 2 and action.shape[0] == 1
		return utils.to_np(action[0])

	def act_noise(self, obs, n, sample=False):
		"""Samples a noisy action from the Actor network

		Args:
			obs (torch.Tensor): non-normed image input
			n (int): the number of elements to 0 out
			sample (bool): True = true sampling, False = deterministic sampling

		Returns:
			np.array version of action vector
		"""
		obs = torch.FloatTensor(obs).to(self.device)
		obs = obs.unsqueeze(0)
		dist = self.actor.noise(obs, n=n)
		action = dist.sample() if sample else dist.mean
		action = action.clamp(*self.action_range)
		assert action.ndim == 2 and action.shape[0] == 1
		return utils.to_np(action[0])

	def update_critic_nstep(self, h, obses, actions, rewards, not_dones, steps, logger, step):
		first_idx = np.random.choice(len(self.levels[h]) - 1)

		action_idxs = [x for x in range(first_idx, steps[first_idx + 1])]

		with torch.no_grad():
			# dist = self.actor(obses[:, steps[1], :, :, :])

			if self.shared_enc:

				dist = self.actor(obses[:, steps[first_idx + 1], :, :, :])

			else:

				obz = torch.cat([
					self.critics[i].encoder(obses[:, steps[first_idx + 1], :, :, :]) for i in range(self.h)
				], dim=1)

				dist = self.actor.forward_fc(obz)


			self.a_std.append(self.actor.outputs['std'].mean().item())

			next_action = dist.rsample()
			log_prog = dist.log_prob(next_action).sum(-1, keepdim=True)
			target_Q1, target_Q2 = self.critic_targets[h](obses[:, steps[first_idx + 1], :, :, :], next_action)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prog

			reward = torch.zeros(rewards[:, 0].shape).to(self.device)
			returns_disc_exp = 0
			for j in action_idxs:
				reward += rewards[:, j] * (self.discount ** returns_disc_exp)
				returns_disc_exp += 1

			target_Q = reward + (not_dones[:, steps[first_idx + 1]] * self.discount ** returns_disc_exp * target_V)

		current_Q1, current_Q2 = self.critics[h](obses[:, first_idx, :, :, :], actions[:, first_idx ])
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		#
		# critic_loss = F.huber_loss(current_Q1, target_Q) + F.huber_loss(current_Q2, target_Q)

		self.critic_losses[h+1].append(critic_loss.item())
		self.eoo_hist.append(not_dones[:, steps[first_idx + 1]].mean().item())
		logger.log('train_critic/loss', critic_loss, step)

		self.critic_optimizers[h].zero_grad()

		critic_loss.backward()

		if self.clip_grad:
			torch.nn.utils.clip_grad_norm_(self.critics[h].parameters(), self.clip_grad)

		g = []
		for p in self.critics[h].encoder.parameters():
			g.extend(p.grad.reshape(-1).cpu().numpy())
		self.critic_grads[h + 1].append(np.sum(np.array(g) ** 2) ** 0.5)

		self.critic_optimizers[h].step()

	def update_actor_and_alpha(self, obs, logger, step):
		"""Performs Actor and alpha update

		Args:
			obs (torch.Tensor): non-normed image input
			logger (logging.Logger): Logger class
			step (int): the step number

		Returns:
			None
		"""

		actor_loss = 0

		if self.shared_enc:

			dist = self.actor(obs, detach_encoder=True)

		else:

			obz = torch.cat([
				self.critics[i].encoder(obs, detach=True) for i in range(self.h)
			], dim=1)

			dist = self.actor.forward_fc(obz)

		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		self.alpha_hist.append(self.alpha.detach().cpu().item())

		for jj in range(self.h):
			# detach conv filters, so we don't update them with the actor loss
			# dist = self.actor(obs, detach_encoder=True)
			actor_Q1, actor_Q2 = self.critics[jj](obs, action, detach_encoder=True)
			actor_Q = torch.min(actor_Q1, actor_Q2)

			actor_loss += (
					self.alpha.detach() * log_prob - actor_Q #* self.discount**self.skips[jj]
			).mean()

		# Need to average?
		# actor_loss /= self.h
		logger.log('train_actor/loss', actor_loss, step)
		logger.log('train_actor/target_entropy', self.target_entropy, step)
		logger.log('train_actor/entropy', -log_prob.mean(), step)

		# optimize the actor
		# In the original DrQ impl, the last dense layer of the encoder is optimized... 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()

		g = []
		for p in self.actor.trunk.parameters():
			g.extend(p.grad.reshape(-1).cpu().numpy())
		self.a_grads.append(np.sum(np.array(g)**2) ** 0.5)

		self.actor_optimizer.step()

		self.actor.log(logger, step)

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
		logger.log('train_alpha/loss', alpha_loss, step)
		logger.log('train_alpha/value', self.alpha, step)
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

	def update_h_sharing(self, replay_buffer):
		for i in range(self.h):
			self.ksls[i].train(True)
		obses, actions, obses_next, rewards, not_dones = replay_buffer.sample_traj_efficient(self.batch_size,
	 																					 self.replay_len)
		for i in reversed(range(self.h)):
			conn_range = [x for x in reversed(range(self.h)) if x > i]

			outs = None

			for j in conn_range:
				_, outs = self.update_h_sharing_layer2(j, obses, actions, rewards, outs, self.levels[j], gather_loss=False)

			loss, _ = self.update_h_sharing_layer2(i, obses, actions, rewards, outs, self.levels[i])

			self.ksl_loss_hist[i + 1].append(loss.item())

			for j in conn_range + [i]:
				self.ksl_optimizers[j].zero_grad()

			loss.backward()

			g = []
			for p in self.ksls[i].encoder_online.parameters():
				g.extend(p.grad.reshape(-1).cpu().numpy())

			self.ksl_grads[i + 1].append(np.sum(np.array(g)**2)**0.5)

			for j in conn_range + [i]:
				self.ksl_optimizers[j].step()

	def update_h_sharing_layer2(self, h, obses, actions, rewards, ins, steps, gather_loss=True):
		z_o = self.ksls[h].encoder_online(obses[:, 0, :, :, :])
		loss = 0

		outs = []
		outs.append(z_o)

		for i, step in enumerate(steps[:-1]):
			action_idxs = [x for x in range(step, steps[i + 1])]

			action_concat = torch.cat([
				actions[:, p] for p in action_idxs
			], dim=1)

			z_m = self.ksls[h].encoder_momentum(obses[:, steps[i + 1], :, :, :]).detach()

			if not ins:
				z_o = self.ksls[h].gru(action_concat, z_o)
				outs.append(z_o)

			else:
				# C(z[h+1] | one-hot)
				position_one_hot = torch.zeros((self.batch_size, len(steps) - 1)).to(self.device)
				position_one_hot[:, i] += 1
				state = self.ksls[h].pre_gru(
					torch.cat([
						torch.cat(ins, dim=1), position_one_hot
					], dim=1)
				)

				z_o = self.ksls[h].gru(action_concat, z_o, state)
				outs.append(z_o)

			if gather_loss:
				if self.repr:
					z_m_hat = self.ksls[h].Wz(z_o)
					loss += self.loss_fn(z_m_hat, z_m).mean()

		return loss, outs

	def compare_grads(self, replay_buffer):
		obses, actions, obses_next, rewards, not_dones = replay_buffer.sample_traj_efficient(self.batch_size,
	 																					 self.replay_len)

		# Perform a forward pass through level 2's stuff
		for _ in range(10):
			outs = None
			loss, outs = self.update_h_sharing_layer2(1, obses, actions, rewards, outs, self.levels[1])
			self.ksl_optimizers[1].zero_grad()
			loss.backward()

			g = []
			for p in self.ksls[1].encoder_online.parameters():
				g.extend(p.grad.reshape(-1).cpu().numpy())

			print(f'Grad norm level 2 from only level 2: {np.sum(np.array(g)**2)**0.5}')

		# Now repeat but also include the
		outs = None
		_, outs = self.update_h_sharing_layer2(1, obses, actions, rewards, outs, self.levels[1], gather_loss=False)
		loss, _ = self.update_h_sharing_layer2(0, obses, actions, rewards, outs, self.levels[0])

		self.ksl_optimizers[1].zero_grad()
		loss.backward()
		g = []
		for p in self.ksls[1].encoder_online.parameters():
			g.extend(p.grad.reshape(-1).cpu().numpy())

		print(f'Grad norm level 2 including level 1: {np.sum(np.array(g) ** 2) ** 0.5}')


	def update(self, replay_buffer, logger, step):
		"""Performs an Actor, alpha, Critic, and KSL update according to the class-speficied frequencies.
		Also, performs EMA updates.

		Args:
			replay_buffer (replay_buffer.ReplayBuffer): the agent's replay buffer
			logger (logging.Logger): Logger class
			step (int): the step number

		Returns:
			None
		"""
		obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
			self.batch_size)

		logger.log('train/batch_reward', reward.mean(), step)

		if step % self.ksl_update_frequency == 0 and self.repr:
			self.update_h_sharing(replay_buffer)

		if self.critic_nstep:
			obses, actions, obses_next, rewards, not_dones = replay_buffer.sample_traj_efficient(self.batch_size,
																								 self.replay_len)
			for i in reversed(range(self.h)):
				self.update_critic_nstep(i, obses, actions, rewards, not_dones, self.levels[i], logger, step)

		if step % self.actor_update_frequency == 0:
			self.update_actor_and_alpha(obs, logger, step)

		if step % self.critic_target_update_frequency == 0:
			for i in range(self.h):
				utils.soft_update_params(self.critics[i].Q1, self.critic_targets[i].Q1, 0.01)
				utils.soft_update_params(self.critics[i].Q2, self.critic_targets[i].Q2, 0.01)
				utils.soft_update_params(self.ksls[i].encoder_online, self.ksls[i].encoder_momentum, 0.05)

		# Check grad magnitudes
		# print(step)
		if step % (1000 // 8) == 0:
			self.compare_grads(replay_buffer)

	def save(self, dir):
		torch.save(
			self.actor.state_dict(), f'{dir}/actor.pt'
		)

		for i in range(self.h):
			torch.save(self.critics[i].state_dict(), f'{dir}/critic_{i}.pt')
			torch.save(self.critic_targets[i].state_dict(), f'{dir}/critic_target_{i}.pt')
			torch.save(self.ksls[i].state_dict(), f'{dir}/ksl_{i}.pt')


	def load(self, dir, extras):
		self.actor.load_state_dict(
			torch.load(dir + extras + '_actor.pt')
		)

		self.critic.load_state_dict(
			torch.load(dir + extras + '_critic.pt')
		)

		self.ksl.load_state_dict(
			torch.load(dir + extras + '_ksl.pt')
		)
