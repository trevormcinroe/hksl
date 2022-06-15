import sys
sys.path.append('../')
sys.path.append('../../')

import time
import numpy as np
import pickle
import dmc2gym
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from distracting_control import suite

torch.backends.cudnn.benchmark = True
import os
here = os.path.dirname(os.path.abspath(__file__))

def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    elif cfg.env == 'cartpole_two_poles':
        domain_name = 'cartpole'
        task_name = 'two_poles'
    elif cfg.env == 'cartpole_three_poles':
        domain_name = 'cartpole'
        task_name = 'three_poles'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_env_gdc(cfg, distractor, train=True):
    """Helper function to create dm_control environment with distractors"""
    _, env = cfg.env.split('-')

    env, difficulty, movement = env.split('__')

    print(f'Making {movement} environment with {distractor} distractions on the {difficulty} setting...')

    if env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name, task_name = env.split('_')

    # The davis_path is not necessary unless you are using the bg videos, which we not used in our study.
    davis_path = '../../../../davis/DAVIS/JPEGImages/480p'

    camera_id = 2 if domain_name == 'quadruped' else 0

    env = suite.load(
        domain_name,
        task_name,
        difficulty,
        background_dataset_path=davis_path,
        dynamic=movement == 'dynamic',
        render_kwargs={'height': 84, 'width': 84, 'camera_id': camera_id},
        background_dataset_videos='train' if train else 'val',
        allow_color_distraction=distractor == 'color',
        allow_background_distraction=False,
        allow_camera_distraction=distractor == 'camera'
    )

    max_episode_steps = (1000 + cfg.action_repeat - 1) // cfg.action_repeat

    env = utils.DCSEnvironment(env, max_episode_steps, cfg.seed, camera_id, cfg.action_repeat)

    env = utils.FrameStack(env, k=cfg.frame_stack)

    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace:
    def __init__(self, cfg):
        self.work_dir = './'
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        if 'gdc' in cfg.env:
            # The 'eval' version of the env is the exact same for all but BG vids...
            self.env = make_env_gdc(cfg)
            self.env_eval = self.env

        else:
            self.env = make_env(cfg)
            self.env_eval = self.env

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        cfg.agent.params.env = cfg.env
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.agent.action_repeat = cfg.action_repeat

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad,
                                          self.device,
                                          self.cfg.env)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        eps_reward = []

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env_eval.reset()
            n_actions = 0

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                    n_actions += 1

                obs, reward, done, info = self.env_eval.step(action)
                self.video_recorder.record(self.env_eval)
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.agent.name}-{self.cfg.env}-s{self.cfg.seed}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes

        sd_episode_reward = np.std(eps_reward)

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

        return average_episode_reward, sd_episode_reward

    def run(self):
        print(f'Eval freq: {self.cfg.eval_frequency}')
        print(f'k: {self.agent.k}')
        print(f'lr: {self.cfg.lr}')
        print(f'Agent: {self.agent.name}')
        print(f'Env: {self.cfg.env}')
        print(f'Seed: {self.cfg.seed}')

        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        eval_mean = []

        while self.step < (self.cfg.num_train_steps // self.cfg.action_repeat):

            if done:

                # Wanna save video?
                # if self.step == (self.cfg.num_train_steps // self.cfg.action_repeat):
                #     self.video_recorder = VideoRecorder(self.work_dir)

                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)

                    means, sds = self.evaluate()
                    eval_mean.append(means)

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0

                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            if done:
                eeo = 1
            else:
                eeo = 0

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max, eeo)

            obs = next_obs
            episode_step += 1
            self.step += 1

        with open(f'./{here}/{self.agent.name}-{self.cfg.env}-s{self.cfg.seed}-b{self.cfg.batch_size}-k{self.cfg.agent.params.k}-h{self.cfg.agent.params.h}-recon{self.agent.recon}-covar{self.agent.covar}-rpred{self.agent.r_pred}-p{self.cfg.p}-factor{self.cfg.factor}-connected{self.agent.connected}-mean-{self.step}.data', 'wb') as f:
            pickle.dump(eval_mean, f)

        # # self.agent.save(
        #     dir='<ENTER/LOCATION/HERE>',
        # )


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W

    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
