import subprocess
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--distractor', default='None')
parser.add_argument('--k', type=int)

args = parser.parse_args()

m = {
	'cartpole_swingup':
		{
			'ar': 8,
			'ef': 1250,
			'lr': 1e-3,
			'ns': 100100
		},
	'reacher_easy':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3
		},
	'ball_in_cup_catch':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3
		},
	'cheetah_run':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3
		},
	'walker_walk':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3
		},
	'finger_spin':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3
		},
	'gdc-cartpole_swingup__easy__dynamic':
		{
			'ar': 8,
			'ef': 1250,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-ball_in_cup_catch__easy__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-cheetah_run__easy__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-reacher_easy__easy__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-walker_walk__easy__dynamic':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-finger_spin__easy__dynamic':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-cartpole_swingup__medium__dynamic':
		{
			'ar': 8,
			'ef': 1250,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-ball_in_cup_catch__medium__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-cheetah_run__medium__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-reacher_easy__medium__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-walker_walk__medium__dynamic':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-finger_spin__medium__dynamic':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-cartpole_swingup__hard__dynamic':
		{
			'ar': 8,
			'ef': 1250,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-ball_in_cup_catch__hard__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-cheetah_run__hard__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-reacher_easy__hard__dynamic':
		{
			'ar': 4,
			'ef': 2500,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-walker_walk__hard__dynamic':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3,
			'ns': 100100
		},
	'gdc-finger_spin__hard__dynamic':
		{
			'ar': 2,
			'ef': 5000,
			'lr': 1e-3,
			'ns': 100100
		}
}


ar = m[args.env]['ar']
ef = m[args.env]['ef']
lr = m[args.env]['lr']
ns = m[args.env]['ns']

seeds = [np.random.randint(1000) for _ in range(1)]

for seed in seeds:
	subprocess.run(
		f'python3 train.py save_video=False agent.class=spr.SPRAgent agent.params.k={args.k} num_train_steps={ns} env={args.env} distractor={args.distractor} lr={lr} action_repeat={ar} seed={seed} eval_frequency={ef} ',
		shell=True
	)
