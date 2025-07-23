Directory structure:
└── using Reinforcement learing/
    ├── approach 1/
    │   ├── config.py
    │   ├── config.yaml
    │   ├── dqn.py
    │   ├── env.py
    │   ├── hangman_agent.py
    │   ├── log.py
    │   ├── memory.py
    │   └── play.py
    └── approach 2/
        ├── Hangman
        └── Word_List


Files Content:

================================================
FILE: using Reinforcement learing/approach 1/config.py
================================================
  
from typing import Any
import re
import yaml
import json


def load_yaml(path):
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)

class TrainingConfig():
    def __init__(self, config:dict=None)-> None:
        self.batch_size = config.get("batch_size")
        self.learning_rate = config.get("learning_rate")
        self.loss = config.get("loss")
        self.num_episodes = config.get("num_episodes")
        self.train_steps = config.get("train_steps")
        self.warmup_episode = config.get("warmup_episode")
        self.save_freq = config.get("save_freq")

class OptimizerConfig():
    def __init__(self, config:dict=None)-> None:
        self.name = config.get("name")
        self.lr_min = config.get("lr_min")
        self.lr_decay = config.get("lr_decay")

class RlConfig():
    def __init__(self, config:dict=None) -> None:
        self.gamma = config.get("gamma")
        self.max_steps_per_episode = config.get("max_steps_per_episode")
        self.target_model_update_episodes = config.get("target_model_update_episodes")
        self.max_queue_length = config.get("max_queue_length")

class EpsilonConfig():
    def __init__(self, config:dict=None) -> None:
        self.max_epsilon = config.get("max_epsilon")
        self.min_epsilon = config.get("min_epsilon")
        self.decay_epsilon = config.get("decay_epsilon")

class Config:
    """ User config class """
    def __init__(self, path: str=None):        
        if path is not None:
            config = load_yaml(path)
            self.training = TrainingConfig(config.get("training", {}))
            self.optimizer = OptimizerConfig(config.get("optimizer", {}))
            self.rl = RlConfig(config.get("rl", {}))
            self.epsilon = EpsilonConfig(config.get("epsilon", {}))


================================================
FILE: using Reinforcement learing/approach 1/config.yaml
================================================
training:
  batch_size: 32
  learning_rate: 0.001
  loss: huber
  num_episodes: 10000
  train_steps: 1000000
  warmup_episode: 10
  save_freq: 1000

optimizer:
  name: adam
  lr_min: 0.0001
  lr_decay: 5000

rl:
  gamma: 0.99
  max_steps_per_episode: 30
  target_model_update_episodes: 100
  max_queue_length: 50000

epsilon:
  max_epsilon: 1
  min_epsilon: 0.1
  decay_epsilon: 400


================================================
FILE: using Reinforcement learing/approach 1/dqn.py
================================================
import logging

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

# create logger
logger = logging.getLogger('root')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = None


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()       
        num_classes = 26
        num_layers = 1
        input_size = 27
        hidden_size = 32
        seq_length = 27
        
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size+26, num_classes) #fully connected 1
        # self.softmax = nn.LogSoftmax(dim=2)
        self.fc = nn.Linear(num_classes, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,word, actions):
        logger.debug("Forwad:word shape = {0}".format(word.shape))
        logger.debug("Forward:actions shape = {0}".format(actions.shape))
        logger.debug("Forward:actions = {0}".format(actions))
        logger.debug("Forward:word = {0}".format(word))
        # actions = torch.tensor(actions.reshape(-1, 26))
        # print(h_0.requires_grad)
        # Propagate input through LSTM
        # print("Forward: word req grad", word.requires_grad)
        output, (hn, cn) = self.lstm(word.float()) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        logger.debug("Forward: hn shape = {0}".format(hn.shape))
        combined = torch.cat((hn, actions), 1)
        out = self.relu(combined)
        logger.debug("Forward: combined shape = {0}".format(combined.shape))
        out = self.fc_1(out) #first Dense
        out = self.fc(out)
        logger.debug("Forward: Out = {0}".format(out))
        # out_binary = out.argmax(1)
        # print("out binary = ", out_binary)
        # final_action = torch.zeros(out.shape).scatter(1, out_binary.unsqueeze (1), 1).long()
        # print("Forward: out = ", out.numpy().tolist())
        # out = self.relu(out) #relu
        # out = self.fc(out) #Final Output
        # out = self.softmax(out)
        # print("Forward: out =", out.argmax())
        return out
    


================================================
FILE: using Reinforcement learing/approach 1/env.py
================================================
import gym
from gym import spaces
import string
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np
from gym.utils import seeding
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

config = None

MAX_WORDLEN = 25

with open("config.yaml", 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

logger = logging.getLogger('root')
# logger.warning('is when this event was logged.')


class HangmanEnv(gym.Env):

	def __init__(self):
		# super().__init__()
		self.vocab_size = 26
		self.mistakes_done = 0
		f = open("./words.txt", 'r').readlines()
		self.wordlist = [w.strip() for w in f]
		self.action_space = spaces.Discrete(26)
		self.vectorizer = CountVectorizer(tokenizer=lambda x: list(x))
		# self.wordlist = [w.strip() for w in f]
		self.vectorizer.fit([string.ascii_lowercase])
		self.config = config
		self.char_to_id = {chr(97+x): x for x in range(self.vocab_size)}
		self.char_to_id['_'] = self.vocab_size
		self.id_to_char = {v:k for k, v in self.char_to_id.items()}
		self.observation_space = spaces.Tuple((
			spaces.MultiDiscrete(np.array([25]*27)),     #Current obscured string
			spaces.MultiDiscrete(np.array([1]*26))      #Actions used                      #Wordlen
		))
		self.observation_space.shape=(27, 26)
		self.seed()

	def filter_and_encode(self, word, vocab_size, min_len, char_to_id):
		"""
		checks if word length is greater than threshold and returns one-hot encoded array along with character sets
		:param word: word string
		:param vocab_size: size of vocabulary (26 in this case)
		:param min_len: word with length less than this is not added to the dataset
		:param char_to_id
		"""

		#don't consider words of lengths below a threshold
		word = word.strip().lower()
		if len(word) < min_len:
			return None, None, None

		encoding = np.zeros((len(word), vocab_size + 1))
		#dict which stores the location at which characters are present
		#e.g. for 'hello', chars = {'h':[0], 'e':[1], 'l':[2,3], 'o':[4]}
		chars = {k: [] for k in range(vocab_size+1)}

		for i, c in enumerate(word):
			idx = char_to_id[c]
			#update chars dict
			chars[idx].append(i)
			#one-hot encode
			encoding[i][idx] = 1

		zero_vec = np.zeros((MAX_WORDLEN - encoding.shape[0], vocab_size + 1))
		encoding = np.concatenate((encoding, zero_vec), axis=0)

		return encoding

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def choose_word(self):
		return random.choice(self.wordlist)

	def count_words(self, word):
		lens = [len(w) for w in self.wordlist]
		counter=dict(collections.Counter(lens))
		return counter[len(word)]

	def reset(self):
		self.mistakes_done = 0
		# inputs, labels, miss_chars, input_lens, status = self.dataloader.return_batch()
		self.word = self.choose_word()
		self.wordlen = len(self.word)
		self.gameover = False
		self.win = False
		self.guess_string = "_"*self.wordlen
		self.actions_used = set()
		self.actions_correct = set()
		logger.info("Reset: Resetting for new word")

		logger.info("Reset: Selected word= {0}".format(self.word))


		self.state = (
			self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
			np.array([0]*26)
		)

		logger.debug("Reset: Init State = {self.state}")

		return self.state

	def vec2letter(self, action):
		letters = string.ascii_lowercase
		# idx = np.argmax(action==1)
		return letters[action]

	def getGuessedWord(self, secretWord, lettersGuessed):
		"""
		secretWord: string, the word the user is guessing
		lettersGuessed: list, what letters have been guessed so far
		returns: string, comprised of letters and underscores that represents
		what letters in secretWord have been guessed so far.
		"""
		secretList = []
		secretString = ''
		for letter in secretWord:
			secretList.append(letter)
		for letter in secretList:
			if letter not in lettersGuessed:
				letter = '_'
			secretString += letter
		return secretString


	def check_guess(self, letter):
		if letter in self.word:
			self.prev_string = self.guess_string
			self.actions_correct.add(letter)
			self.guess_string = self.getGuessedWord(self.word, self.actions_correct)
			return True
		else:
			return False

	def step(self, action):
		done = False
		reward = 0
		if string.ascii_lowercase[action.argmax()] in self.actions_used:
			reward = -4.0
			self.mistakes_done += 1
			logger.info("Env Step: repeated action, action was= {0}".format(string.ascii_lowercase[action.argmax()]))
			logger.info("ENV STEP: Mistakes done = {0}".format(self.mistakes_done))
			if self.mistakes_done >= 6:
				done = True
				self.win = True
				self.gameover = True
		elif string.ascii_lowercase[action.argmax()] in self.actions_correct:
			reward = -3.0
			logger.info("Env Step: repeated correct action, action was= {0}".format(string.ascii_lowercase[action.argmax()]))
			logger.info("ENV STEP: Mistakes done = {0}".format(self.mistakes_done))
			# done = True
			# self.win = True
			# self.gameover = True
		elif self.check_guess(self.vec2letter(action.argmax())):
			logger.info("ENV STEP: Correct guess, evaluating reward, guess was = {0}".format(string.ascii_lowercase[action.argmax()]))
			if(set(self.word) == self.actions_correct):
				reward = 10.0
				done = True
				self.win = True
				self.gameover = True
				logger.info("ENV STEP: Won Game, evaluating reward, guess was = {0}".format(string.ascii_lowercase[action.argmax()]))
			# self.evaluate_subset(action)
			reward = +1.0
			# reward = self.edit_distance(self.state, self.prev_string)
			self.actions_correct.add(string.ascii_lowercase[action.argmax()])
		else:
			logger.info("ENV STEP: Incorrect guess, evaluating reward, guess was = {0}".format(string.ascii_lowercase[action.argmax()]))
			self.mistakes_done += 1
			if(self.mistakes_done >= 6):
				reward = -5.0
				done = True
				self.gameover = True
			else:
				reward = -2.0

		self.actions_used.add(string.ascii_lowercase[action.argmax()])

		logger.info("ENV STEP: actions used = {0}".format(" ".join(self.actions_used)))
		logger.info("ENV STEP: actions used = {0}".format(" ".join(self.actions_used)))
		self.state = (
			self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
			self.vectorizer.transform(list(self.actions_used)).toarray()[0]
		)
		logger.debug("Intermediate State = {self.state}")
		return (self.state, reward, done, {'win' :self.win, 'gameover':self.gameover})



================================================
FILE: using Reinforcement learing/approach 1/hangman_agent.py
================================================
from re import T
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from torch.cuda import init
from env import HangmanEnv
from torch.autograd import Variable
from config import Config
import yaml
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from memory import Transition, ReplayMemory
from dqn import DQN
from log import setup_custom_logger
import time
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

# create logger
logger = setup_custom_logger('root', "./latest.log", "INFO")
logger.propagate = False
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = None
        
class HangmanPlayer():
    def __init__(self, env, config):
        self.memory = None
        self.steps_done = 0
        self.episode_durations = []
        self.last_episode = 0
        self.reward_in_episode = []
        self.env = env
        self.id = int(time.time())
        self.config = config
        self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compile()
        
    def compile(self):     
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # summary(self.target_net, (128, 25, 27))
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
    def _update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def _adjust_learning_rate(self, episode):
        delta = self.config.training.learning_rate - self.config.optimizer.lr_min
        base = self.config.optimizer.lr_min
        rate = self.config.optimizer.lr_decay
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    def _get_action_for_state(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("This value is ", policy_net(state).max(1))
                selected_action = self.policy_net(torch.tensor(state[0]), torch.tensor([state[1]])).argmax()
                # print("ActionSelect: require grad = ", tens.requires_grad)
                value = selected_action.numpy()
                b = np.zeros((value.size, 26))
                # print("Value = ", value)
                b[np.arange(value.size), value] = 1
                selected_action = torch.from_numpy(b).long()
                # print("from net, got action = ", int(value))
                # final_action = torch.zeros(26).scatter(1, selected_action.unsqueeze (1), 1).long()
                # print("Final action2 = ", selected_action)
                return selected_action
        else:
            a = np.array(random.randrange(self.n_actions))
            b = np.zeros((1, 26))
            b[0, a] = 1
            print(b.shape)
            selected_action = torch.from_numpy(b).long()
            # print("ActionSelect: action selected = ", type(random.randrange(self.n_actions)))
            # final_action = torch.zeros(26).scatter(1, selected_action.unsqueeze (1), 1).long()
            # print("Final action = ", selected_action)
            return selected_action
        
    def save(self):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "reward_in_episode": self.reward_in_episode,
            "episode_durations": self.episode_durations,
            "config": self.config
            }, f"./models/pytorch_{self.id}.pt")
        
    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(torch.tensor([state], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.tensor([next_state], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))
    
    def fit(self):
        num_episodes = 5000000
        self.memory = ReplayMemory(10000)
        self.episode_durations = []
        self.reward_in_episode = []
        reward_in_episode = 0
        self.epsilon_vec = []
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            # self.check_grad()
            # last_screen = get_screen()
            # current_screen = get_screen()
            # print("Fit: state = ", state)
            state = (state[0].reshape(-1, 25, 27), state[1])
            # state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                # self.check_grad()
                action = self._get_action_for_state(state)
                # self.check_grad()
                # print("Fit: action = ", action.shape)
                next_state, reward, done, info = self.env.step(action)
                
                # reward = torch.tensor([reward], device=device)
                next_state = (next_state[0].reshape(-1, 25, 27), next_state[1])        # Observe new state
                # action_vector = next_state[1]
                # print("Fit: next state actions = ", next_state[1])
                # last_state = state
                # state=next_state
                # current_screen = get_screen()
                # if not done:
                #     next_state = current_screen - last_screen
                # else:
                #     next_state = None

                # Store the transition in memory
                self._remember(state[0], next_state[1], next_state[0], reward, done)
                
                if i_episode >= self.config.training.warmup_episode:
                    self._train_model()
                    self._adjust_learning_rate(i_episode - self.config.training.warmup_episode + 1)
                    done = (t == self.config.rl.max_steps_per_episode - 1) or done
                else:
                    done = (t == 5 * self.config.rl.max_steps_per_episode - 1) or done
                
                # Move to the next state
                state = next_state
                reward_in_episode += reward

                if done:
                    self.episode_durations.append(t + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    # self.epsilon_vec.append(epsilon)
                    reward_in_episode = 0
                    break

                # Update the target network, copying all weights and biases in DQN
                if i_episode % 50 == 0:
                    self._update_target()

                # if i_episode % self.config.training.save_freq == 0:
                #     self.save()

                self.last_episode = i_episode
                # Move to the next state

                # Perform one step of the optimization (on the policy network)
                if done:
                    self.episode_durations.append(t + 1)
                    # plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.config.rl.target_model_update_episodes == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            if i_episode % self.config.training.save_freq == 0:
                self.save()
    
    def _train_model(self):  
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        
        

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_batch.resize_(BATCH_SIZE, 25, 27)
        non_final_next_states.resize_(BATCH_SIZE, 25, 27)
        # print("action batch = ", action_batch.numpy().sum())
        # action_batch.resize_(BATCH_SIZE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print("Here printing = ", self.policy_net(state_batch, action_batch))
        state_action_values = self.policy_net(state_batch, action_batch).gather(0, action_batch)
        # print("State batch = ", state_batch)
        # print("action batch = ", action_batch.shape)
        # print("reward batch = ", reward_batch.shape)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" self.target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float)
        temp = self.target_net(non_final_next_states, action_batch).max(1)[0].detach()
        # print("TrainModel: temp = ", temp.numpy().tolist())
        # print(next_state_values)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, action_batch).max(1)[0].detach()
        # Compute the expected Q values
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # print("TrainModel: expected state action values = ", expected_state_action_values)
        # print("TrainModel: state action values = ", state_action_values.numpy().tolist())
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # clone = expected_state_action_values.clone()
        # with torch.enable_grad():
        # print(next_state_values.shape)
        # print(state_action_values.shape)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)).float()
        # print(expected_state_action_values.requires_grad)
        logger.info("trainmodel: loss = {0}".format(loss))
        # loss.requires_grad = True
        self.optimizer.zero_grad()
        # loss = Variable(loss, requires_grad = True)
        # Optimize the model
        # loss.retain_grad()
        # print(loss.grad)
        loss.backward()

        # print("TrainModel: params = ", list(self.policy_net.parameters()))
        for name, param in self.policy_net.named_parameters():
            # print("Param = ", name, param.is_leaf)
            # try:
            # param.is_leaf
            param.grad.data.clamp_(-1, 1)
            # break
            # except:
            #     print("Failed")
            #     # pass
        self.optimizer.step()
        
    def play(self, verbose:bool=False, sleep:float=0.2, max_steps:int=100):
        # Play an episode
        try:
            actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

            iteration = 0
            state = self.env.reset()  # reset environment to a new, random state
            state = (state[0].reshape(-1, 25, 27), state[1])
            if verbose:
                print(f"Iter: {iteration} - Action: *** - Reward ***")
            time.sleep(sleep)
            done = False

            while not done:
                action = self._get_action_for_state(state)
                iteration += 1
                state, reward, done, info = self.env.step(action)
                display.clear_output(wait=True)
                self.env.render()
                if verbose:
                    print(f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}")
                time.sleep(sleep)
                if iteration == max_steps:
                    print("cannot converge :(")
                    break
        except KeyboardInterrupt:
            pass
            
    def evaluate(self, max_steps:int=100):
        try:
            total_steps, total_penalties = 0, 0
            episodes = 100

            for episode in trange(episodes):
                state = self.env.reset()  # reset environment to a new, random state
                state = (state[0].reshape(-1, 25, 27), state[1])
                nb_steps, penalties, reward = 0, 0, 0

                done = False

                while not done:
                    action = self._get_action_for_state(state)
                    state, reward, done, info = self.env.step(action)

                    if reward == -10:
                        penalties += 1

                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True

                total_penalties += penalties
                total_steps += nb_steps

            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")    
        except KeyboardInterrupt:
            pass


================================================
FILE: using Reinforcement learing/approach 1/log.py
================================================
import logging

def setup_custom_logger(handle_name, filename, level):
    # logging.basicConfig(filename=filename, encoding='utf-8', level=level, filemode="w")
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    handler = logging.FileHandler('latest_logs.log', mode="w")
    logging.StreamHandler(stream=None)
    handler.setFormatter(formatter)

    logger = logging.getLogger(handle_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


================================================
FILE: using Reinforcement learing/approach 1/memory.py
================================================
import logging
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.rng = np.random.default_rng()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.choice(np.arange(len(self.memory)), batch_size, replace=False)
        res = []
        for i in idx:
            res.append(self.memory[i])
        return res
        # return self.rng.choice(self.memory, batch_size, replace=False)

    def __len__(self):
        return len(self.memory)


================================================
FILE: using Reinforcement learing/approach 1/play.py
================================================
import gym
from env import HangmanEnv
import time
import random
import numpy as np
import pickle
import string



N_EPISODES = 100000
epsilon = 0.1
alpha = 0.1
gamma = 0.8

q_table = {}

games_won = 0
games_played = 0

try:
    with open('q_table.pickle', 'rb') as handle:
        q_table = pickle.load(handle)
  
except:
    print("Unable to find file")

env = HangmanEnv()

for i_episode in range(N_EPISODES):
    state = env.reset()
    games_played += 1
    print("Starting episode")
    if state not in q_table:
        q_table[state] = np.random.rand(26)
    while(1):
        print(state)
        # action = env.action_space.sample()
        # temp = np.zeroes(26)
        
        if random.uniform(0, 1) < epsilon:
            print("Exploring")
            # temp = [0]*26
            # temp[random.randint(0, 25)] = 1
            action = random.randint(0, 25)
        else:
            print("Taking best action")
            action = np.argmax(q_table[state])
        print("action = ", string.ascii_lowercase[action])
        next_state, reward, done, info = env.step(action)
        
        print("reward = ", reward)
        if next_state not in q_table:
            q_table[next_state] = np.random.rand(26)
        
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        print("Old val = ", old_value)
        print("next max = ", next_max)
        
        new_value = (1 - alpha)*old_value + alpha * (reward + gamma*next_max)
        q_table[state][action] = new_value
        
        print("new val = ", new_value)
        # print("next max = ", next_max)
        
        state = next_state
        
        if done:
            print("Episode finished", info)
            if info['win'] == True:
                games_won += 1
            break
        # print(q_table)
        # time.sleep(0.1)
        

env.close()

try:
    q_file = open('q_table.pickle', 'wb')
    pickle.dump(q_table, q_file)
    q_file.close()
  
except:
    print("Something went wrong")
    
    
print("Win %:", games_won/games_played)


================================================
FILE: using Reinforcement learing/approach 2/Hangman
================================================
import random
import matplotlib.pyplot as plt
import numpy as np
from word_lists import get_hangman_words
from collections import Counter, defaultdict
from statistics import mean
from sklearn.model_selection import train_test_split

# Function to calculate letter frequency in the word list
def calculate_letter_frequency(word_list):
    all_letters = "".join(word_list)  # Concatenating all the words
    letter_counts = Counter(all_letters)  # Count frequency of each letter
    total_letters = sum(letter_counts.values())  # Calculate total number of letters
    
    # Calculate letter frequencies and sort them
    sorted_counts = letter_counts.most_common()
    letter_frequency = {k: v / total_letters for k, v in sorted_counts}
    return letter_frequency


# Function to decide the next action (guessing a letter) based on the current state and epsilon-greedy strategy
def choose_action(state, letter_frequency, epsilon=0.1):
    global action_count
    action_count += 1  # Increase action count for each guess
    vowels = set("aeiou")
    consonants = set("bcdfghjklmnpqrstvwxyz")
    # Extract 25% of most frequent consonants
    top_consonants = [k for k, v in sorted(letter_frequency.items(), key=lambda item: item[1], reverse=True) if k in consonants][:len(consonants)//4]
    # Initialize
    if state in Q_table and np.random.rand() > epsilon:
        max_q_value = max(Q_table[state].values())
        max_actions = [action for action, q_value in Q_table[state].items() if q_value == max_q_value]
        best_action = max(max_actions, key=lambda action: letter_frequency.get(action, 0))
    else:
        remaining_letters = set("abcdefghijklmnopqrstuvwxyz") - set(state[2])
        best_action = random.choice(list(remaining_letters))
    # First two guesses should be vowels
    if action_count == 1 or action_count == 2:
        remaining_vowels = vowels - set(state[2])
        if remaining_vowels:
            best_action = random.choice(list(remaining_vowels))
    # Third and fourth should be top 25% frequent consonants
    elif action_count == 3 or action_count == 4:
        remaining_top_consonants = set(top_consonants) - set(state[2])
        if remaining_top_consonants:
            best_action = random.choice(list(remaining_top_consonants))
    return best_action


def encode_state(state, attempts_left, guessed_letters):
    return (state, attempts_left, "".join(sorted(guessed_letters)))


# Function to update the Q-table based on Q-learning algorithm
def update_Q_table(state, action, reward, new_state, Q_table, epoch, alpha_initial=0.5, decay_factor=0.01, gamma=0.9):
    alpha = alpha_initial / (1 + decay_factor * epoch)
    max_future_Q = max(Q_table[new_state].values()) if new_state in Q_table else 0
    current_Q = Q_table[state][action]

    Q_table[state][action] = (1 - alpha) * current_Q + alpha * (reward + gamma * max_future_Q)


# Hangman game logic
def hangman(word, letter_frequency, epochs):
    global action_count
    action_count = 0
    state_str = "_" * len(word)  # Initial state with underscores
    attempts_left = 6  # Number of attempts left
    guessed_letters = set()  # Letters already guessed

    state = encode_state(state_str, attempts_left, guessed_letters)
    
    history = []  # Store actions history

    while attempts_left > 0 and "_" in state_str:
        action = choose_action(state, letter_frequency)  # Choose next letter to guess
        
        guessed_letters.add(action)
        new_state_str = state_str  # Initialize new state to be same as old state initially

        # If chosen letter is in the word, reveal it
        if action in word:
            for i, c in enumerate(word):
                if c == action:
                    new_state_str = new_state_str[:i] + c + new_state_str[i+1:]
        
        else:  # Otherwise, reduce attempts
            attempts_left -= 1

        new_state = encode_state(new_state_str, attempts_left, guessed_letters)
        
        if action in word:
            reward = 0.3  # Positive reward for correct guess
        else:
            reward = -0.1  # Negative reward for wrong guess

        if "_" not in new_state_str:
            reward += 1  # Bonus reward for winning the game

        update_Q_table(state, action, reward, new_state, Q_table, epochs) # Update Q-table

        state_str = new_state_str  # Set new state as the current state
        state = new_state  # Update the entire state
        history.append(action)  # Add action to history

    return state_str, history  # Return final state and action history


def train_model(word_list, letter_frequency, epochs, epsilon_decay=0.995):
    epsilon = 1.0  # Start with a high epsilon
    win_ratio = []
    for epoch in range(epochs):
        win_count = 0
        epsilon *= epsilon_decay  # Reduce epsilon each epoch
        for word in word_list:
            result,_ = hangman(word, letter_frequency, epochs)
            if "_" not in result:  # A win if no underscores are left in the result
                win_count +=1
                win_ratio.append(win_ratio)
    return win_ratio

def evaluate_test_words(custom_words, letter_frequency):
    print("\nEvaluating on custom words:")
    wins = 0  # Number of games won
    total_games = len(custom_words)  # Total number of games    
    for word in custom_words:
        result, _ = hangman(word, letter_frequency, epochs=1)
        win = "_" not in result  # A win if no underscores are left in the result
        print(f"Word: {word}, Guess: {result} Result: {'Win' if win else 'Lose'}")
        
        if win:
            wins += 1
    win_percentage = (wins / total_games) * 100  # Calculate win percentage
    print(f"Custom Words - Wins: {wins}, Total: {total_games}, Win Percentage: {round(win_percentage, 3)}%")


# Main function to run the experiment
def run_experiment(runs):
    word_list = get_hangman_words()
    letter_frequency = calculate_letter_frequency(word_list)
    win_ratios = []
    try:
        for i in range(runs):
            print(f"Run {i + 1}: Training...")
            win_ratio = train_model(word_list, letter_frequency, epochs)
            print(f"Win ratio for run {i + 1}: {win_ratio}")

    except KeyboardInterrupt:
        print("\nExperiment interrupted. Calculating win ratio.")

    finally:
        mean_win_ratio = round(mean(win_ratios), 3)
        print(f"Mean Win Ratio: {mean_win_ratio}")

    # Custom Test Words
    test_words = ["ocean", "hangman", "rocket", "python", "cheese", "caramel", "part", "letter", "candle", "series"]
    evaluate_test_words(test_words, letter_frequency)

# Run the experiment
epochs = 40
runs = 1000
action_count = 0
word_list = get_hangman_words()
Q_table = defaultdict(lambda: defaultdict(float))
run_experiment(runs)



================================================
FILE: using Reinforcement learing/approach 2/Word_List
================================================
# word_lists.py

def get_hangman_words():
    word_list = [
        'apple', 'orange', 'grape', 'mango', 'pear', 'plum', 'quince', 'berry', 'cherry',
        'melon', 'date', 'lemon', 'lime', 'peach', 'apricot', 'coconut', 'fig', 'guava',
        'kiwi', 'papaya', 'almond', 'pecan', 'walnut', 'cashew', 'tomato', 'potato',
        'onion', 'carrot', 'radish', 'beet', 'celery', 'pepper', 'yam', 'turnip',
        'cabbage', 'lettuce', 'spinach', 'kale', 'ginger', 'garlic', 'zucchini',
        'cucumber', 'olive', 'avocado', 'banana', 'pineapple', 'raisin', 'peanut',
        'chestnut', 'hazelnut', 'pumpkin', 'squash', 'blueberry', 'cranberry', 'strawberry',
        'artichoke', 'eggplant', 'broccoli', 'cauliflower', 'asparagus', 'parsley',
        'coriander', 'oregano', 'thyme', 'rosemary', 'tarragon', 'sage', 'chives', 'mint',
        'dill', 'cumin', 'curry', 'turmeric', 'paprika', 'saffron', 'cayenne', 'anise',
        'cardamom', 'cloves', 'nutmeg', 'cinnamon', 'vanilla', 'allspice', 'fennel',
        'mustard', 'peppercorn', 'sesame', 'poppy', 'caraway', 'juniper', 'chili',
        'wasabi', 'horseradish', 'capers', 'cilantro', 'basil', 'leek', 'scallion',
        'shallot', 'parsnip', 'rhubarb', 'bamboo', 'watercress', 'sprouts', 'arugula', 
        'tiger', 'zebra', 'koala', 'badger', 'otter', 'beaver', 'weasel', 'wombat',
        'gopher', 'ferret', 'skunk', 'shrew', 'ocelot', 'jaguar', 'panther', 'cougar',
        'leopard', 'cheetah', 'hyena', 'lynx', 'moose', 'elk', 'reindeer', 'gazelle',
        'impala', 'buffalo', 'ostrich', 'peacock', 'pelican', 'seagull', 'sparrow',
        'pigeon', 'parrot', 'penguin', 'eagle', 'hawk', 'falcon', 'raven', 'swan',
        'goose', 'crane', 'heron', 'osprey', 'kestrel', 'vulture', 'condor', 'grouse',
        'puffin', 'cormorant', 'ibis', 'dove', 'emu', 'kite', 'owl', 'quetzal',
        'finch', 'robin', 'tern', 'thrush', 'toucan', 'warbler', 'wren', 'starling',
        'swallow', 'oriole', 'crow', 'jay', 'lark', 'loon', 'merlin', 'plover', 'lapwing',
        'kingfish', 'avocet', 'murre', 'pochard', 'sandpiper', 'snipe', 'stilt', 'grebe',
        'tropic', 'shearwater', 'petrel', 'coot', 'bittern', 'hobby', 'curlew', 'egret',
        'flamingo', 'gannet', 'godwit', 'guillemot', 'harrier', 'jaeger', 
        'magpie',"hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine", "neon",
        "sodium", "magnesium", "aluminum", "silicon", "phosphorus", "sulfur", "chlorine", "argon", "potassium", "calcium",
        "scandium", "titanium", "vanadium", "chromium", "manganese", "iron", "cobalt", "nickel", "copper", "zinc",
        "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton", "rubidium", "strontium", "yttrium", "zirconium",
        "niobium", "molybdenum", "technetium", "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin",
        "antimony", "tellurium", "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
        "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium", "thulium", "ytterbium",
        "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium", "iridium", "platinum", "gold", "mercury",
        "thallium", "lead", "bismuth", "polonium", "astatine", "radon", "francium", "radium", "actinium", "thorium",
        "protactinium", "uranium", "neptunium", "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
        "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium", "hassium", "meitnerium", "darmstadtium",
        "roentgenium", "copernicium", "nihonium", "flerovium", "moscovium", "livermorium", "tennessine", "oganesson",
        "people", "world", "school", "program", "problem", "family", 
        "health", "service", "change", "nation", "create", "public", "follow", 
        "during", "present", "without", "office", "point", "client", "become", 
        "product", "project", "matter", "person", "policy", "result", "report", 
        "figure", "friend", "begin", "design", "always", "several", "growth", 
        "nature", "around", "provide", "control", "teacher", "develop", "effect", 
        "return", "research", "company", "market", "though", "thanks", "specific", 
        "society", "general", "subject", "history", "picture", "similar", "degree", 
        "action", "accept", "almost", "enough", "website", "explain", "require", 
        "entire", "career", "rather", "regard", "choice", "future", "medical", 
        "single", "chance", "theory", "within", "window", "answer", "increase", 
        "farmer", "leader", "remain", "system", "success", "season", "purpose", 
        "either", "recent", "reflect", "discuss", "agreed", "method", "beyond", 
        "happen", "animal", "factor", "decide", "indeed", "inside", "nearly", 
        "energy", "create", "amount", "change", "follow", "parent", "impact", "source", "charge", "corner",
        "record", "summer", "police", "either", "remain", "modern", "budget", "treaty", "gender",
        "debate", "animal", "factor", "object", "attack", "victim", "writer", "client", "extend",
        "relief", "oppose", "status", "survey", "policy", "series", "ruling", "entire", "sample",
        "define", "safety", "estate", "singer", "remove", "return", "thanks", "reduce", "visual",
        "focus", "wealth", "solely", "discus", "affair", "scheme", "union", "cycle", "whole",
        "grant", "daily", "theory", "option", "listen", "friend", "demand", "search", "marker",
        "employ", "source", "aspect", "impact", "obtain", "notion", "access", "credit", "nature",
        "aspect", "stress", "finish", "resolve", "unique", "target", "signal", "relate", "select",
        "gather", "manage", "derive", "proven", "mostly", "advise", "reveal", "random", "couple",
        "always", "switch", "charge", "expose", "giving", "player", "easier", "enrich", "mobile",
        "activity", "approach", "business", "campaign", "complete", "describe", "determine",
        "economic", "evidence", "generate", "hospital", "identify", "increase", "indicate",
        "industry", "interest", "involved", "military", "personal", "physical", "planning",
        "position", "positive", "practice", "pressure", "previous", "problems", "progress",
        "purchase", "question", "research", "resource", "response", "security", "sentence",
        "specific", "standard", "strategy", "strength", "supplies", "thoughts", "thousand",
        "together", "training", "violence", "whatever", "wildlife", "analysis", "argument",
        "awareness", "beneficial", "capacity", "cultural", "disaster", "function", "graphics",
        "hardware", "instance", "judgment", "language", "medicine", "numerous", "operator",
        "priority", "rational", "relevant", "reliable", "sentence", "struggle", "survival",
        "thinking", "ultimate", "variable", "building", "criteria", "database", "domestic",
        "feedback", "guidance", "incident", "location", "magnitude", "narrative", "overcome",
        "paradigm", "regional", "scenario", "teaching", "temporal", "academic", "accurate",
        "boundary", "commodity", "consumer", "contrast", "critical", "currency", "equation",
        "freedom", "passage", "justice", "ethical", "respect", "journey", "wisdom", "reality",
        "courage", "inspire", "qualify", "discuss", "compose", "fitness", "finance", "proceed",
        "explore", "convert", "kingdom", "provoke", "sustain", "chamber", "isolate", "fulfill",
        "empower", "refresh", "tension", "liberty", "diverse", "inquire", "perform", "predict",
        "acquire", "absolve", "achieve", "decline", "dismiss", "endure", "safety", "secure",
        "enable", "engage", "enrich", "deploy", "derive", "scheme", "evoke", "orient", "emerge",
        "permit", "expose", "purify", "refine", "verify", "reject", "invoke", "retain", "impose",
        "inform", "assert", "clarify", "compare", "combat", "empire", "excuse", "output", "renew",
        "rescue", "revive", "unveil", "utilize", "subdue", "unfold", "uphold", "attend", "awake",
        "unfold", "withstand", "approve", "arrive", "assist", "assume", "assure", "attract",
        "augment", "bewild", "comply", "concur", "defer", "denote", "deploy", "derive", "detect",
        "devote", "differ", "disarm", "equate", "induce", "invoke", "launch", "omit", "persist",
        "prosper", "pursue", "revoke", "unite", "uplift", "vex", "yearn", "allure", "beseech",
        "bestow", "bewail", "blight", "blithe", "boast", "baffle", "chide", "clasp", "cling",
        "coax", "crave", "deplore", "despise", "dread", "entreat", "fawn", "flee", "gloat", "grieve",
        "grasp", "inquire", "lament", "loathe", "mourn", "recoil", "shun", "snub", "spurn", "wail",
        "believe", "examine", "insight", "neutral", "pioneer", "prosper", "recruit", "venture",
        "afflict", "compile", "endorse", "foresee", "inhabit", "nourish", "outlive", "persist",
        "project", "restore", "sustain", "traverse", "balance", "compose", "disclose", "enforce",
        "fracture", "insulate", "obstruct", "override", "persist", "reclaim", "repress", "suspend",
        "transmit", "acclaim", "blossom", "consume", "divulge", "execute", "generate", "involve",
        "occupy", "pollute", "rebuild", "resolve", "seclude", "swindle", "uncover", "appease",
        "clarify", "conflict", "distort", "exclude", "forfeit", "inhibit", "neglect", "oppress",
        "provoke", "redeem", "refrain", "suppress", "thrive", "withhold", "advocate", "betray",
        "condemn", "disrupt", "elude", "forsake", "hinder", "migrate", "obscure", "prohibit",
        "revoke", "sabotage", "tamper", "thwart", "withdraw", "coexist", "conform", "deceive",
        "delight", "endure", "enhance", "fulfill", "harvest", "maintain", "overcome", "prevail",
        "restrain", "scarcity", "treasure", "undermine", "convince", "decorate", "embrace",
        "fascinate", "illumine", "manifest", "overwhelm", "perceive", "reverber", "simulate",
        "tolerate", "acknowledge", "alienate", "cultivate", "dominate", "enlighten", "flourish",
        "integrate", "liberate", "orchestrate", "perpetuate", "radiate", "reconcile", "subjugate",
        "achieve", "adapt", "adjust", "advise", "analyze", "assure", "attract", "benefit", "compose",
        "concede", "condone", "console", "construct", "consume", "contribute", "convict", "debate",
        "decide", "deduct", "defer", "define", "defy", "deliver", "demand", "deny", "depict",
        "derive", "design", "detect", "deter", "determine", "devise", "dictate", "differ", "diminish",
        "discern", "dispute", "dissolve", "distill", "divert", "divulge", "dominate", "donate",
        "doubt", "draft", "dwell", "earn", "eject", "elapse", "elicit", "embody", "embrace", "emit",
        "enable", "enact", "endure", "enforce", "engage", "enhance", "enlist", "enroll", "entail",
        "entice", "equal", "equip", "erode", "evade", "evoke", "evolve", "exceed", "exclude",
        "exert", "exhibit", "exist", "expand", "expect", "expel", "expire", "explain", "explore",
        "export", "expose", "extract", "fathom", "favor", "fear", "feel", "fend", "fetch", "find",
        "flaunt", "flee", "flinch", "flock", "flow", "fluctuate", "foil", "forbid", "force",
        "forecast", "forge", "form", "foster", "found", "gain", "gather", "gauge", "generate",
        "govern", "grant", "greet", "grow", "halt", "handle", "happen", "harbor", "harm", "hatch",
        "haunt", "head", "heal", "heap", "hear", "heat", "help", "hide", "highlight", "hinder",
        "hint", "hire", "hoard", "hold", "hone", "hook", "hope", "hover", "hunt", "hurry"]

    unique_set = set()
    unique_list = []
    duplicates = []

    for item in word_list:
        if item not in unique_set:
            unique_set.add(item)
            unique_list.append(item)
        else:
            duplicates.append(item)

    #print(f"Duplicates: {duplicates}")
    #print(f"Unique list: {unique_list}")

    return unique_list



