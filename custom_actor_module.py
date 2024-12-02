# Let us start our tutorial by importing some useful stuff.

# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# Useful classes:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline

# And a couple external libraries:
import numpy as np
import os


# Maximum number of training 'epochs':
# (training is checkpointed at the end of each 'epoch', this is also when training metrics can be logged to wandb)
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused, waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb credentials:
# (Change this with your own if you want to keep your training curves private)
# (Also, please use your own wandb account if you are going to log huge stuff :) )

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
# name of the wandb project in which your run will appear
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

# this line sets your wandb API key as the active key
os.environ['WANDB_API_KEY'] = wandb_key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
# (In TMRL, networking is managed by tlspyo. The following are tlspyo parameters.)
# IP of the machine running the Server (trainer point of view)
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
# IP of the machine running the Server (worker point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
# when training over the Internet, it is safer to change this to "TLS"
security = cfg.SECURITY
# (please read the security instructions on GitHub)


# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

# Sample preprocessor for data augmentation:
sample_preprocessor = None

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH

# Preprocessor applied by the worker to the observations it collects:
# (Note: if your script defines the name "obs_preprocessor", we will use your preprocessor instead of the default)
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)
device_worker = 'cpu'


# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Whether you are using grayscale (default) or color images:
# (Note: The tutorial will stop working if you use colors)
img_grayscale = cfg.GRAYSCALE

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN

# Number of actions in the action buffer (this is part of observations):
# (Note: The tutorial will stop working if you change this)
act_buf_len = cfg.ACT_BUF_LEN


memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)


# Let us import the ActorModule that we are supposed to implement.
# We will use PyTorch in this tutorial.
# TMRL readily provides a PyTorch-specific subclass of ActorModule:
from tmrl.actor import TorchActorModule

# Plus a couple useful imports:
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor


# Here is the MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# The next utility computes the dimensionality of CNN feature maps when flattened together:
def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# The next utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (
        conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (
        conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class VanillaCNN(nn.Module):
    def __init__(self, q_net=False):
        super(VanillaCNN, self).__init__()
        self.q_net = q_net

        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(
            self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(
            self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(
            self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(
            self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out

    def forward(self, images):  # Simplified forward pass
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.flat_features)
        return x


class IQN(nn.Module):
    def __init__(self, layer_type="ff"):
        super(IQN, self).__init__()
        self.input_shape = 137
        self.action_size = 12
        self.K = 32
        self.N = 8
        self.n_cos = 64
        self.layer_size = 256
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(
            1, 1, self.n_cos)  # Starting from 0 as in the paper

        self.head = nn.Linear(
            self.input_shape, self.layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, self.layer_size)
        self.ff_1 = nn.Linear(self.layer_size, self.layer_size)
        self.ff_2 = nn.Linear(self.layer_size, self.action_size)
        self.cnn = VanillaCNN()
        #weight_init([self.head_1, self.ff_1])

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(
            batch_size, n_tau).unsqueeze(-1)  # (batch_size, n_tau, 1)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (
            batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def forward(self, input, num_tau=8):

        speed, gear, rpm, images, act1, act2 = input

        cnn_features = self.cnn(images)

        x = torch.cat((speed, gear, rpm, cnn_features, act1, act2), -1)
        batch_size = x.shape[0]


        x = torch.relu(self.head(x))

        # cos shape (batch, num_tau, layer_size)
        cos, taus = self.calc_cos(batch_size, num_tau)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(
            batch_size, num_tau, self.layer_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)


        x = torch.relu(self.ff_1(x))
        x = self.ff_2(x)


        return x.view(batch_size, num_tau, self.action_size), taus

    def get_action(self, inputs):
        quantiles, _ = self.forward(inputs, self.K)
        actions = quantiles.mean(dim=1)
        return actions


import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class MyActorModule(TorchActorModule):

    def __init__(self, observation_space, action_space):

        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)

        # Our hybrid CNN+MLP policy:
        self.net = IQN()
        self.actions = np.array([[[x, y, z]] for x in [0, 1] for y in [0, 1] for z in [-1, 0, 1]])

    def save(self, path):

        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):

        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def forward(self, input):
        return self.net.forward(input)

    def act(self, obs, test=False):
        with torch.no_grad():
            action_values = self.net.get_action(obs)
            action = np.argmax(action_values.cpu().data.numpy())
            a = self.actions[action]
            # a = a / np.sum(np.abs(a))
            a = a.squeeze(0)
            return a


from tmrl.training import TrainingAgent

# We will also use a couple utilities, and the Adam optimizer:

from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import torch.optim as optim
import random


class DQN_Agent(TrainingAgent):
    """Interacts with and learns from the environment."""


    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=MyActorModule,
                 state_size=137,
                 action_size=12,
                 layer_size=256,
                 n_step=1,
                 BATCH_SIZE=256,
                 LR=1e-3,
                 TAU=1e-2,
                 GAMMA=0.99,
                 UPDATE_EVERY=1):

        # required arguments passed to the superclass:
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        self.state_size = state_size
        self.action_size = action_size
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step

        self.action_step = 4
        self.last_action = None

        # IQN-Network
        self.qnetwork_local = model_cls(
            observation_space, action_space)
        self.qnetwork_target = no_grad(deepcopy(self.qnetwork_local))

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.actions = np.array([[[x, y, z]] for x in [0, 1] for y in [0, 1] for z in [-1, 0, 1]])


        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def get_actor(self):

        return self.qnetwork_local

    def calculate_huber_loss(self, td_errors, k=1.0):

        loss = torch.where(td_errors.abs() <= k, 0.5 *
                           td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (
            td_errors.shape[0], 8, 8), "huber loss has wrong shape"
        return loss

    def get_index(self, row, lookup):
        for i, item in enumerate(lookup):
            if np.array_equal(row, item[0]):
                return i
        return -1


    def train(self, experiences):

        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, _ = experiences
        actions_ = np.array([int(self.get_index(row, self.actions)) for row in actions])
        actions_ = actions_.reshape(-1, 1) #Reshape to column vector
        actions_ = torch.tensor(actions_, dtype=torch.int64)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target.forward(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1)  # (batch_size, 1, N)

        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(
            2, actions_.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (
            self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = self.calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        # , keepdim=True if per weights get multipl
        loss = quantil_l.sum(dim=1).mean(dim=1)
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        ret_dict = dict(
            loss=loss.detach().cpu().item()
        )
        return ret_dict

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)


training_agent_cls = partial(
    DQN_Agent,
    model_cls=MyActorModule,
    state_size=137,
    action_size=12,
    layer_size=256,
    n_step=1,
    BATCH_SIZE=256,
    LR=1e-3,
    TAU=1e-2,
    GAMMA=0.99,
    UPDATE_EVERY=1)


training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true',
                        help='launches the server')
    parser.add_argument('--trainer', action='store_true',
                        help='launches the trainer')
    parser.add_argument('--worker', action='store_true',
                        help='launches a rollout worker')
    parser.add_argument('--test', action='store_true',
                        help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run()

        # Note: if you want to log training metrics to wandb, replace my_trainer.run() with:
        # my_trainer.run_with_wandb(entity=wandb_entity,
        #                           project=wandb_project,
        #                           run_id=wandb_run_id)

    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=MyActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run(test_episode_interval=10)
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)
