Directory structure:
└── using RNN/
    ├── README.md
    ├── config.yaml
    ├── dataloader.py
    ├── game.py
    ├── generic_model.py
    ├── model.py
    └── train_test.py


Files Content:

================================================
FILE: using RNN/README.md
================================================
# Introduction

Hangman is a popular word game. Person 1 chooses a word at his will and the goal of Person 2 is to predict the word by predicting one character at a time. At every step, Person 1 gives the following feedback to Person 2 depending on the predicted character:
1. If the character is present, Person 1 informs Person 2 of all the positions at which the character is present.
e.g. if the word is 'hello' and Person 2 predicts 'e', Person 1 has to report that the character 'e' is present at position 2
2. If the character is not present, Person 1 simply reports that the character is not present in his/her word.

Refer to [this](https://en.wikipedia.org/wiki/Hangman_(game)) Wikipedia article for more details.

# Model

1. The main model consists of a RNN which acts as an encoder. It takes the encoded version of the incomplete string and returns the hidden state at the last time step of the last layer.
2. On the other hand, we take the missed characters, encode it into a vector and pass it through a linear layer.
3. The outputs from 1 and 2 are concatenated and we pass this concatenated vector through a hidden layer and map it to a final layer with number of neurons = size of vocabulary (26 in our case for the English alphabet).

The model is trained using Binary Cross Entropy Loss since it is a multi-label classification problem.
A pretrained model can be found [here](https://drive.google.com/open?id=1hVBlS3zxTqcVktVZEHTv2ivg-oKpr-KQ).
It is a 2-layer, 512 hidden unit GRU with dropout and trained using Adam.

# Dataset

The model is trained on a corpus of around 227k English words which can be found in the datasets folder. Testing is done on a corpus of 20k words. There is a 50% overlap between the training dataset and the testing dataset.


# Performance

After preliminary testing, here is a plot of the average number of misses vs length of the word:
![Performance](https://github.com/methi1999/hangman/blob/master/imgs/performance.png)
As you can observe, the average misses decreases as length of the word decreases. This makes sense intuitively since longer the word -> chances of higher number of unique characters increases -> chances of a predicted chaarcter not being present decreases.

# Further Work

~~1. Weight examples during training with weights inversely proportional to length.~~


================================================
FILE: using RNN/config.yaml
================================================
dataset: 'dataset/' # dataset directory
models: 'models/' # for storing models
plots: 'plots/' # for plots
pickle: 'pickle/' # pickle dumps root path

cuda: True #whether to use NVIDIA cuda
    
test_per_epoch: 0 #test per epoch i.e. how many times in ONE epoch
test_every_epoch: 50 #after how many epochs
print_per_epoch: 3 #print loss function how often in each epoch
save_every: 400 #save models after how many epochs
plot_every: 25 #save plots for test loss/train loss/accuracy

resume: True #resume training from saved model

lr: 0.0005 #learning rate

drop_uniform: False #whether dropping of character sets is independent of set size
reset_after: 400 #generate a new random dataset after these manh epochs
vocab_size: 26 #size of vocabulary. 26 engliush letters in our case
min_len: 3 #words with length less than min_len are not added to the dataset

rnn: 'GRU' #type of RNN. Can be LSTM/GRU
use_embedding: True #whether to use character embeddings
embedding_dim: 128 #if use_embedding, dimension of embedding
hidden_dim: 512 #hidden dimension of RNN
output_mid_features: 256 #number of neurons in hidden layer after RNN
miss_linear_dim: 256 #miss chars are projected to this dimension using a simple linear layer
num_layers: 2 #number of layers in RNN
dropout: 0.3 #dropout
batch_size: 4000 #batch size for training and testing
epochs: 3000 #total no. of epochs to train



================================================
FILE: using RNN/dataloader.py
================================================
import numpy as np
import random
import os
import pickle
import json

np.random.seed(7)

#number of dimensions in input tensor over the vocab size
#1 in this case which represents the blank character
extra_vocab = 1

def filter_and_encode(word, vocab_size, min_len, char_to_id):
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

	encoding = np.zeros((len(word), vocab_size + extra_vocab))
	#dict which stores the location at which characters are present
	#e.g. for 'hello', chars = {'h':[0], 'e':[1], 'l':[2,3], 'o':[4]}
	chars = {k: [] for k in range(vocab_size)}

	for i, c in enumerate(word):
		idx = char_to_id[c]
		#update chars dict
		chars[idx].append(i)
		#one-hot encode
		encoding[i][idx] = 1

	return encoding, [x for x in chars.values() if len(x)], set(list(word))


def batchify_words(batch, vocab_size, using_embedding):
	"""
	converts a list of words into a batch by padding them to a fixed length array
	:param batch: a list of words encoded using filter_and_encode function
	:param: size of vocabulary (26 in our case)
	:param: use_embedding: if True, 
	"""

	total_seq = len(batch)
	if using_embedding:
		#word is a list of indices e.g. 'abd' will be [0,1,3]
		max_len = max([len(x) for x in batch])
		final_batch = []

		for word in batch:
			if max_len != len(word):
				#for index = vocab_size, the embedding gives a 0s vector
				zero_vec = vocab_size*np.ones((max_len - word.shape[0]))
				word = np.concatenate((word, zero_vec), axis=0)
			final_batch.append(word)

		return np.array(final_batch)
	else:
		max_len = max([x.shape[0] for x in batch])
		final_batch = []

		for word in batch:
			#word is a one-hot encoded array of dimensions length x vocab_size
			if max_len != word.shape[0]:
				zero_vec = np.zeros((max_len - word.shape[0], vocab_size + extra_vocab))
				word = np.concatenate((word, zero_vec), axis=0)
			final_batch.append(word)

		return np.array(final_batch)


def encoded_to_string(encoded, target, missed, encoded_len, char_to_id, use_embedding):
	"""
	convert an encoded input-output pair back into a string so that we can observe the input into the model
	encoded: array of dimensions padded_word_length x vocab_size
	target: 1 x vocab_size array with 1s at indices wherever character is present
	missed: 1 x vocav_size array with 1s at indices wherever a character which is NOT in the word, is present
	encoded_len: length of word. Needed to retrieve the original word from the padded word
	char_to_id: dict which maps characters to ids
	use_embedding: if character embeddings are used
	"""

	#get reverse mapping
	id_to_char = {v:k for k, v in char_to_id.items()}

	if use_embedding:
		word = [id_to_char[x] if x < len(char_to_id) - 1 else '*' for x in list(encoded[:encoded_len])]
	else:
		word = [id_to_char[x] if x < len(char_to_id) - 1 else '*' for x in list(np.argmax(encoded[:encoded_len, :], axis=1))]

	word = ''.join(word)
	target = [id_to_char[x] for x in list(np.where(target != 0)[0])]
	missed = [id_to_char[x] for x in list(np.where(missed != 0)[0])]
	print("Word, target and missed characters:", word, target, missed)

#class which constructs database and returns batches during training/testing
class dataloader:

	def __init__(self, mode, config):

		self.mode = mode
		
		self.vocab_size = config['vocab_size']
		#blank vec is the one-hot encoded vector for unknown characters in the word
		self.blank_vec = np.zeros((1, self.vocab_size + extra_vocab))
		self.blank_vec[0, self.vocab_size] = 1
		
		self.batch_size = config['batch_size']
		self.total_epochs = config['epochs']

		#char_to_id is done specifically for letters a-z
		self.char_to_id = {chr(97+x): x for x in range(self.vocab_size)}
		self.char_to_id['BLANK'] = self.vocab_size
		self.id_to_char = {v:k for k, v in self.char_to_id.items()}
		
		self.drop_uniform = config['drop_uniform']
		self.use_embedding = config['use_embedding']

		#dump mapping so that all modules use the same mapping
		if self.mode == 'train':
			with open(config['pickle']+'char_to_id.json', 'w') as f:
				json.dump(self.char_to_id, f)

		#dataset for training and testing
		if mode == 'train':
			filename = config['dataset'] + "250k.txt"
		else:
			filename = config['dataset'] + "20k.txt"

		#if already dumped, load the database from dumped pickle file
		pkl_path = config['pickle'] + mode + '_input_dump.pkl'
		if os.path.exists(pkl_path):
			with open(pkl_path, 'rb') as f:
				self.final_encoded = pickle.load(f)
		else:
			corpus = []
			print("Processing dataset for", self.mode)
			#read .txt file
			with open(filename, 'r') as f:
				corpus = f.readlines()

			self.final_encoded = []
			
			for i, word in enumerate(corpus):
				#print progress
				if i%(len(corpus)//10) == len(corpus)//10-1:
					print("Done:", i+1, "/", len(corpus))
			
				encoding, unique_pos, chars = filter_and_encode(word, self.vocab_size, config['min_len'], self.char_to_id)
				if encoding is not None: #indicates that word length is above threshold
					self.final_encoded.append((encoding, unique_pos, chars))

			#dump encoded database 
			with open(pkl_path, 'wb') as f:
				pickle.dump(self.final_encoded, f)

		#construct input-output pairs
		self.refresh_data(0)

	def refresh_data(self, epoch):
		"""
		constructs a database from the corpus
		each training example consists of 3 main tensors:
		1. encoded word with blanks which represent unknown characters
		2. labels which corresponds to a vector of dimension vocab_size with 1s at indices where characters are to be predicted
		3. miss_chars is a vector of dimension vocab_size with 1s at indcies which indicate that the character is NOT present
		   this information is gained from feedback received from the game i.e. if we predict 'a' and the game returns that 'a' is not present, we update this vector
		   and aske the model to predict again
		"""

		print("Refreshing data")

		#the probability with which we drop characters. As training progresses, the probability increases and
		#hence we feed the model with words which have fewer exisitng characters and more blanks -> makes it more challenging to predict
		drop_prob = 1/(1+np.exp(-epoch/self.total_epochs))
		self.cur_epoch_data = []
		all_chars = list(self.char_to_id.keys())
		all_chars.remove('BLANK')
		all_chars = set(all_chars)

		for i, (word, unique_pos, chars) in enumerate(self.final_encoded):
			#word is encoded vector of dimensions depending on whether we are useimg_embedding or not
			#unique pos is a list of lists which indicates positions of the letters e.g. for 'hello', unique_pos = [[0], [1], [2,3], [4]]
			#chars is a list of characters present in the word. We take it's complement (where all_chars is the sample space)
			#missed chars are randomly chosen from this complement set

			#how many characters to drop
			num_to_drop = np.random.binomial(len(unique_pos), drop_prob)
			if num_to_drop == 0: #drop atleast 1 character
				num_to_drop = 1

			#whether character sets are chosen uniformly or with prob. inversely proportional to number of occurences of each character
			if self.drop_uniform:
				to_drop = np.random.choice(len(unique_pos), num_to_drop, replace=False)
			else:
				prob = [1/len(x) for x in unique_pos]
				prob_norm = [x/sum(prob) for x in prob]
				to_drop = np.random.choice(len(unique_pos), num_to_drop, p=prob_norm, replace=False)

			#positions of characters to drop
			#e.g. word is 'hello', unique_pos = [[0], [1], [2,3], [4]] and to_drop = [[0], [2,3]]
			#then, drop_idx = [0,2,3]
			drop_idx = []
			for char_group in to_drop:
				drop_idx += unique_pos[char_group]
			
			#since we are dropping these characters, it becomes the target for our model
			#note that if a character is repeated, np.sum will give number_of_occurences at that index. We clip it to 1 since loss expects either 0 or 1
			target = np.clip(np.sum(word[drop_idx], axis=0), 0, 1)

			#making sure that a blank character is not a target
			assert(target[self.vocab_size] == 0) 
			target = target[:-1] # drop blank phone
			
			#drop characters and assign blank_character
			input_vec = np.copy(word)
			input_vec[drop_idx] = self.blank_vec

			#if using embedding, we need to provide character id instead of 1-hot encoded vector
			if self.use_embedding:
				input_vec = np.argmax(input_vec, axis=1)
			
			#randomly pick a few characters from vocabulary as characters which were predicted but declared as not present by game
			not_present = np.array(sorted(list(all_chars - chars)))
			num_misses = np.random.randint(0, 10) #10 because most games end before 10 misses
			miss_chars = np.random.choice(not_present, num_misses)
			miss_chars = list(set([self.char_to_id[x] for x in miss_chars]))
			#e.g. word is 'hello', num_misses = 2, miss_chars [1, 3] (which correspond to the characters b and d)
			
			miss_vec = np.zeros((self.vocab_size))
			miss_vec[miss_chars] = 1
			
			#append tuple to list
			self.cur_epoch_data.append((input_vec, target, miss_vec))

		#shuffle dataset before feeding batches to the model
		np.random.shuffle(self.cur_epoch_data)
		self.num_egs = len(self.cur_epoch_data)
		self.idx = 0

	def return_batch(self):
		"""
		returns a batch for trianing/testing
		"""

		cur_batch = self.cur_epoch_data[self.idx: self.idx+self.batch_size]
		#convert to numoy arrays
		lens = np.array([len(x[0]) for x in cur_batch])
		inputs = batchify_words([x[0] for x in cur_batch], self.vocab_size, self.use_embedding)
		labels = np.array([x[1] for x in cur_batch])
		miss_chars = np.array([x[2] for x in cur_batch])

		self.idx += self.batch_size

		if self.idx >= self.num_egs - 1: #indicates end of epoch
			self.idx = 0
			return inputs, labels, miss_chars, lens, 1
		else:
			return inputs, labels, miss_chars, lens, 0

	def __len__(self):

		return len(self.cur_epoch_data)//self.batch_size


if __name__ == '__main__':		

	import yaml
	with open("config.yaml", 'r') as stream:
		try:
			config = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)

	a = dataloader('test', config)
	c = a.return_batch()
	print(c)



================================================
FILE: using RNN/game.py
================================================
import numpy as np
import json
from train_test import dl_model
import matplotlib.pyplot as plt

#class responsible for actual gameplay
class Game:

    def __init__(self, char_to_id_pth):

        #load character to id mapping dumped by dataloader
        with open(char_to_id_pth, 'r') as f:
            self.char_to_id = json.load(f)

        #initialise model
        self.dl_model = dl_model('test_one')

    def play(self):

        #store correctly and incorresclty predicted characters
        misses, hits = [], []
        print("Instructions: For every predicted letter,")
        print("1. Type 'x' if letter does not exists in your word.")
        print("2. If it does exist, input the places at which it exists.")
        print("For e.g., if the word is hello and model's prediction is 'l', type 3 4 in the prompt. If prediction is 'a', type x.\n")

        print("Think of a word and input the number of characters:")
        num_chars = int(input())
        #initialise target string with blanks
        predicted_string = ['*']*num_chars
        
        while 1:

            #get sorted predictions according to probability
            best_chars = self.dl_model.predict(predicted_string, misses, self.char_to_id)
            #best char is the one with highest probability AND which does not belong to list of incorrectly predicted chars
            #AND not already present in the target string
            for pred in best_chars:
                if pred not in misses and pred not in predicted_string:
                    best_char = pred
                    break
            
            #predict and ask user for feedback
            print("Prediction: " + best_char + "\nWhat do ya think?")

            while 1:
                #get input
                inp = input().strip()
                if inp == 'x': #denotes character not present
                    output = 0
                    break
                try:
                    #if it is present, user returns a list of indices at which character is present (note that indexing begins from 1 for user feedback)
                    output = [int(x) for x in inp.split(' ')]
                    #update target string
                    for pred in output:
                        predicted_string[pred-1] = best_char
                    break
                except:
                    print("Invalid format. Please refer to instructions.") 
                    continue


            if output == 0: #indicates miss
                print("Miss")
                #append to missed characters list
                misses.append(best_char)
            else:
                #correctly predicted
                if '*' in predicted_string: #indicates game is not yet over since we still have unknown characters in target string
                    print("Guess correct! New target: " + ''.join(predicted_string))
                    hits.append(best_char)
                else: #indicates game is over. Report number of misses and return
                    print("Game over. Predicted word: " + ''.join(predicted_string))
                    print("Total misses: ", len(misses))
                    return misses

    def test_performance(self, dataset_pth='dataset/20k.txt', num_trials=100, min_word_len=3, plot=True):

        with open(dataset_pth, 'r') as f:
            words = f.readlines()

        words = [x.strip() for x in words if len(x) >= min_word_len + 1] #+1 since /n is yet to be stripped
        #randomly choose words from corpus
        to_test = np.random.choice(words, num_trials)
        print("Testing performance on the following words:", to_test)

        #stores information about average misses for various lengths of target words
        len_misses_dict = {}
        
        for word in to_test:
            #intialise dict
            if len(word) not in len_misses_dict:
                len_misses_dict[len(word)] = {'misses': 0, 'num': 0}

            hits, misses = [], []
            predicted_string = ['*']*len(word)

            #keep predicting
            while 1:
                #get sorted predictions according to probability
                best_chars = self.dl_model.predict(predicted_string, misses, self.char_to_id)
                #best char is the one with highest probability AND which does not belong to list of incorrectly predicted chars
                #AND not already present in the target string
                for pred in best_chars:
                    if pred not in misses and pred not in predicted_string:
                        best_char = pred
                        break

                found_char = False
                if best_char in word: #denotes character not present
                    #if it is present, user returns a list of indices at which character is present (note that indexing begins from 1 for user feedback)
                    indices = []
                    for i, c in enumerate(word):
                        if c == best_char:
                            indices.append(i)
                    #update target string
                    for pred in indices:
                        predicted_string[pred] = best_char
                    found_char = True

                if found_char is False: #indicates miss
                    #append to missed characters list
                    misses.append(best_char)
                else:
                    #correctly predicted
                    if '*' in predicted_string: #indicates game is not yet over since we still have unknown characters in target string
                        hits.append(best_char)
                    else: #indicates game is over. Report number of misses and return
                        len_misses_dict[len(word)]['misses'] += len(misses)
                        len_misses_dict[len(word)]['num'] += 1
                        break

        len_misses_list = [(l, x['misses']/x['num']) for l, x in len_misses_dict.items()]
        len_misses_list = sorted(len_misses_list, key = lambda x: x[0])
        print("Average number of misses:", len_misses_list)

        #plot performance
        if plot:
            plt.bar([x[0] for x in len_misses_list], [x[1] for x in len_misses_list])
            plt.xlabel('Length of word')
            plt.ylabel('Average misses (lesser the better)')
            plt.title("Comparing performance as a function of word length")
            plt.xticks(list(range(min_word_len, len_misses_list[-1][0])))
            plt.show()


a = Game('pickle/char_to_id.json')
# a.play()
a.test_performance()


================================================
FILE: using RNN/generic_model.py
================================================
import torch
import torch.nn as nn
import os


class generic_model(nn.Module):
    """
    contains basic functions for storing and loading a model
    """

    def __init__(self, config):

        super(generic_model, self).__init__()

        self.config_file = config

    def loss(self, predicted, truth):

        return self.loss_func(predicted, truth)

    # save model, along with loss details and testing accuracy
    # best is the model which has the lowest test loss. This model is used during feature extraction
    def save_model(self, is_best, epoch, train_loss, test_loss, rnn_name, layers, hidden_dim):

        base_path = self.config_file['models']
        if is_best:
            filename = base_path + 'best_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.pth'
        else:
            filename = base_path + str(epoch) + '_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, filename)

        print("Saved model")

    # Loads saved model for resuming training or inference
    def load_model(self, mode, rnn_name, layers, hidden_dim, epoch=None):

        # if epoch is given, load that particular model, else load the model with name 'best'
        if mode == 'test' or mode == 'test_one':

            try:
                if epoch is None:
                    filename = self.config_file['models'] + 'best_' + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.pth'
                else:
                    filename = self.config_file['models'] + str(epoch) + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.pth'

                checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
                # load model parameters
                print("Loading:", filename)
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded pretrained model from:", filename)

            except:
                print("Couldn't find model for testing")
                exit(0)
        # train
        else:
            # if epoch is given, load that particular model else, load the model trained on the most number of epochs
            # e.g. if dir has 400, 500, 600, it will load 600.pth
            if epoch is not None:
                filename = self.config_file['models'] + str(epoch) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.pth'
            else:
                directory = [x.split('_') for x in os.listdir(self.config_file['models'])]
                to_check = []
                for poss in directory:
                    try:
                        to_check.append(int(poss[0]))
                    except:
                        continue

                if len(to_check) == 0:
                    print("No pretrained model found")
                    return 0, [], []
                # model trained on the most epochs
                filename = self.config_file['models'] + str(max(to_check)) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.pth'

            # load model parameters and return training/testing loss and testing accuracy
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print("Loaded pretrained model from:", filename)

            return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['test_loss']



================================================
FILE: using RNN/model.py
================================================
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from generic_model import generic_model


#generic model contains generic methods for loading and storing a model
class RNN(generic_model):

    def __init__(self, config):

        super(RNN, self).__init__(config)

        # Store important parameters
        self.rnn_name = config['rnn']
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim'] 
        self.num_layers = config['num_layers']
        self.embed_dim = config['embedding_dim']
        self.output_dim = config['vocab_size']

        #whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        else:
            self.use_embedding = False
            
        #linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim*2
        mid_features = config['output_mid_features']
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)

        #linear layer after missed characters
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])        

        #declare RNN
        if self.rnn_name == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               dropout=config['dropout'],
                               bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              dropout=config['dropout'],
                              bidirectional=True, batch_first=True)

        #optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def forward(self, x, x_lens, miss_chars):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, input_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :param miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
        :return: tensor of shape (batch size, max sequence length, output dim)
        """        
        if self.use_embedding:
            x = self.embedding(x)
            
        batch_size, seq_len, _ = x.size()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # now run through RNN
        output, hidden = self.rnn(x)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)

        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        #project miss_chars onto a higher dimension
        miss_chars = self.miss_linear(miss_chars)
        #concatenate RNN output and miss chars
        concatenated = torch.cat((hidden, miss_chars), dim=1)
        #predict
        return self.linear2_out(self.relu(self.linear1_out(concatenated)))

    def calculate_loss(self, model_out, labels, input_lens, miss_chars, use_cuda):
        """
        :param model_out: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, vocab_size). 1 at index i indicates that ith character should be predicted
        :param: miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
							passed here to check if model's output probability of missed_chars is decreasing
        """
        outputs = nn.functional.log_softmax(model_out, dim=1)
        #calculate model output loss for miss characters
        miss_penalty = torch.sum(outputs*miss_chars, dim=(0,1))/outputs.shape[0]
        
        input_lens = input_lens.float()
        #weights per example is inversely proportional to length of word
        #this is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lens)/torch.sum(1/input_lens).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        #resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        if use_cuda:
        	weights = weights.cuda()
        
        #actual loss
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_func(model_out, labels)
        return actual_penalty, miss_penalty
        


================================================
FILE: using RNN/train_test.py
================================================
"""
The main driver file responsible for training, testing and predicting
"""

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pickle
from model import RNN
from dataloader import dataloader
from dataloader import encoded_to_string

#load config file
with open("config.yaml", 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

#class responsible for training, testing and inference
class dl_model():

	def __init__(self, mode):

		# Read config fielewhich contains parameters
		self.config = config
		self.mode = mode

		# Architecture name decides prefix for storing models and plots
		feature_dim = self.config['vocab_size']
		self.arch_name = '_'.join(
			[self.config['rnn'], str(self.config['num_layers']), str(self.config['hidden_dim']), str(feature_dim)])

		print("Architecture:", self.arch_name)
		# Change paths for storing models
		self.config['models'] = self.config['models'].split('/')[0] + '_' + self.arch_name + '/'
		self.config['plots'] = self.config['plots'].split('/')[0] + '_' + self.arch_name + '/'

		# Make folders if DNE
		if not os.path.exists(self.config['models']):
			os.mkdir(self.config['models'])
		if not os.path.exists(self.config['plots']):
			os.mkdir(self.config['plots'])
		if not os.path.exists(self.config['pickle']):
			os.mkdir(self.config['pickle'])

		self.cuda = (self.config['cuda'] and torch.cuda.is_available())

		# load/initialise metrics to be stored and load model
		if mode == 'train' or mode == 'test':

			self.plots_dir = self.config['plots']
			# store hyperparameters
			self.total_epochs = self.config['epochs']
			self.test_every = self.config['test_every_epoch']
			self.test_per = self.config['test_per_epoch']
			self.print_per = self.config['print_per_epoch']
			self.save_every = self.config['save_every']
			self.plot_every = self.config['plot_every']

			# dataloader which returns batches of data
			self.train_loader = dataloader('train', self.config)
			self.test_loader = dataloader('test', self.config)
			#declare model
			self.model = RNN(self.config)

			self.start_epoch = 1
			self.edit_dist = []
			self.train_losses, self.test_losses = [], []

		else:

			self.model = RNN(self.config)

		if self.cuda:
			self.model.cuda()

		# resume training from some stored model
		if self.mode == 'train' and self.config['resume']:
			self.start_epoch, self.train_losses, self.test_losses = self.model.load_model(mode, self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
			self.start_epoch += 1

		# load best model for testing/inference
		elif self.mode == 'test' or mode == 'test_one':
			self.model.load_model(mode, self.config['rnn'], self.model.num_layers, self.model.hidden_dim)

		#whether using embeddings
		if self.config['use_embedding']:
			self.use_embedding = True
		else:
			self.use_embedding = False

	# Train the model
	def train(self):

		print("Starting training at t =", datetime.datetime.now())
		print('Batches per epoch:', len(self.train_loader))
		self.model.train()

		# when to print losses during the epoch
		print_range = list(np.linspace(0, len(self.train_loader), self.print_per + 2, dtype=np.uint32)[1:-1])
		if self.test_per == 0:
			test_range = []
		else:
			test_range = list(np.linspace(0, len(self.train_loader), self.test_per + 2, dtype=np.uint32)[1:-1])

		for epoch in range(self.start_epoch, self.total_epochs + 1):

			try:

				print("Epoch:", str(epoch))
				epoch_loss = 0.0
				# i used for monitoring batch and printing loss, etc.
				i = 0

				while True:

					i += 1

					# Get batch of inputs, labels, missed_chars and lengths along with status (when to end epoch)
					inputs, labels, miss_chars, input_lens, status = self.train_loader.return_batch()

					if self.use_embedding:
						inputs = torch.from_numpy(inputs).long() #embeddings should be of dtype long
					else:
						inputs = torch.from_numpy(inputs).float()

					#convert to torch tensors
					labels = torch.from_numpy(labels).float()
					miss_chars = torch.from_numpy(miss_chars).float()
					input_lens = torch.from_numpy(input_lens).long()

					if self.cuda:
						inputs = inputs.cuda()
						labels = labels.cuda()
						miss_chars = miss_chars.cuda()
						input_lens = input_lens.cuda()

					# zero the parameter gradients
					self.model.optimizer.zero_grad()
					# forward + backward + optimize
					outputs = self.model(inputs, input_lens, miss_chars)
					loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
					loss.backward()

					# clip gradient
					# torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
					self.model.optimizer.step()

					# store loss
					epoch_loss += loss.item()

					# print loss
					if i in print_range and epoch == 1:
						print('After %i batches, Current Loss = %.7f' % (i, epoch_loss / i))
					elif i in print_range and epoch > 1:
						print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f, Miss Loss = %.7f' % (
								i, epoch_loss / i, np.mean(np.array([x[0] for x in self.train_losses])), miss_penalty))

					# test model periodically
					if i in test_range:
						self.test(epoch)
						self.model.train()

					# Reached end of dataset
					if status == 1:
						break

				#refresh dataset i.e. generate a new dataset from corpurs
				if epoch % self.config['reset_after'] == 0:
					self.train_loader.refresh_data(epoch)

				#take the last example from the epoch and print the incomplete word, target characters and missed characters
				random_eg = min(np.random.randint(self.train_loader.batch_size), inputs.shape[0]-1)
				encoded_to_string(inputs.cpu().numpy()[random_eg], labels.cpu().numpy()[random_eg], miss_chars.cpu().numpy()[random_eg],
								  input_lens.cpu().numpy()[random_eg], self.train_loader.char_to_id, self.use_embedding)

				# Store tuple of training loss and epoch number
				self.train_losses.append((epoch_loss / len(self.train_loader), epoch))

				# save model
				if epoch % self.save_every == 0:
					self.model.save_model(False, epoch, self.train_losses, self.test_losses,
										  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

				# test every 5 epochs in the beginning and then every fixed no of epochs specified in config file
				# useful to see how loss stabilises in the beginning
				if epoch % 5 == 0 and epoch < self.test_every:
					self.test(epoch)
					self.model.train()
				elif epoch % self.test_every == 0:
					self.test(epoch)
					self.model.train()
				# plot loss and accuracy
				if epoch % self.plot_every == 0:
					self.plot_loss_acc(epoch)

			except KeyboardInterrupt:
				#save model before exiting
				print("Saving model before quitting")
				self.model.save_model(False, epoch-1, self.train_losses, self.test_losses,
									  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
				exit(0)


	# test model
	def test(self, epoch=None):

		self.model.eval()

		print("Testing...")
		print('Total batches:', len(self.test_loader))
		test_loss = 0

		#generate a new dataset form corpus
		self.test_loader.refresh_data(epoch)

		with torch.no_grad():

			while True:

				# Get batch of input, labels, missed characters and lengths along with status (when to end epoch)
				inputs, labels, miss_chars, input_lens, status = self.test_loader.return_batch()
				
				if self.use_embedding:
					inputs = torch.from_numpy(inputs).long()
				else:
					inputs = torch.from_numpy(inputs).float()

				labels = torch.from_numpy(labels).float()
				miss_chars = torch.from_numpy(miss_chars).float()
				input_lens= torch.from_numpy(input_lens).long()

				if self.cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()
					miss_chars = miss_chars.cuda()
					input_lens = input_lens.cuda()

				# zero the parameter gradients
				self.model.optimizer.zero_grad()
				# forward + backward + optimize
				outputs = self.model(inputs, input_lens, miss_chars)
				loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
				test_loss += loss.item()

				# Reached end of dataset
				if status == 1:
					break

		#take a random example from the epoch and print the incomplete word, target characters and missed characters
		#min since the last batch may not be of length batch_size
		random_eg = min(np.random.randint(self.train_loader.batch_size), inputs.shape[0]-1)
		encoded_to_string(inputs.cpu().numpy()[random_eg], labels.cpu().numpy()[random_eg], miss_chars.cpu().numpy()[random_eg],
			input_lens.cpu().numpy()[random_eg], self.train_loader.char_to_id, self.use_embedding)

		# Average out the losses and edit distance
		test_loss /= len(self.test_loader)

		print("Test Loss: %.7f, Miss Penalty: %.7f" % (test_loss, miss_penalty))

		# Store in lists for keeping track of model performance
		self.test_losses.append((test_loss, epoch))

		# if testing loss is minimum, store it as the 'best.pth' model, which is used during inference
		# store only when doing train/test together i.e. mode is train
		if test_loss == min([x[0] for x in self.test_losses]) and self.mode == 'train':
			print("Best new model found!")
			self.model.save_model(True, epoch, self.train_losses, self.test_losses,
								  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

		return test_loss

	def predict(self, string, misses, char_to_id):
		"""
		called during inference
		:param string: word with predicted characters and blanks at remaining places
		:param misses: list of characters which were predicted but game feedback indicated that they are not present
		:param char_to_id: mapping from characters to id
		"""

		id_to_char = {v:k for k,v in char_to_id.items()}

		#convert string into desired input tensor
		if self.use_embedding:
			encoded = np.zeros((len(char_to_id)))
			for i, c in enumerate(string):
				if c == '*':
					encoded[i] = len(id_to_char) - 1 
				else:
					encoded[i] = char_to_id[c]

			inputs = np.array(encoded)[None, :]
			inputs = torch.from_numpy(inputs).long()

		else:

			encoded = np.zeros((len(string), len(char_to_id)))
			for i, c in enumerate(string):
				if c == '*':
					encoded[i][len(id_to_char) - 1] = 1
				else:
					encoded[i][char_to_id[c]] = 1

			inputs = np.array(encoded)[None, :, :]
			inputs = torch.from_numpy(inputs).float()

		#encode the missed characters
		miss_encoded = np.zeros((len(char_to_id) - 1))
		for c in misses:
			miss_encoded[char_to_id[c]] = 1
		miss_encoded = np.array(miss_encoded)[None, :]
		miss_encoded = torch.from_numpy(miss_encoded).float()

		input_lens = np.array([len(string)])
		input_lens= torch.from_numpy(input_lens).long()	

		#pass through model
		output = self.model(inputs, input_lens, miss_encoded).detach().cpu().numpy()[0]

		#sort predictions
		sorted_predictions = np.argsort(output)[::-1]
		
		#we cannnot consider only the argmax since a missed character may also get assigned a high probability
		#in case of a well-trained model, we shouldn't observe this
		return [id_to_char[x] for x in sorted_predictions]

	def plot_loss_acc(self, epoch):
		"""
		take train/test loss and test accuracy input and plot it over time
		:param epoch: to track performance across epochs
		"""

		plt.clf()
		fig, ax1 = plt.subplots()

		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax1.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], color='r', label='Train Loss')
		ax1.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], color='b', label='Test Loss')
		ax1.tick_params(axis='y')
		ax1.legend(loc='upper left')

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.grid(True)
		plt.legend()
		plt.title(self.arch_name)

		filename = self.plots_dir + 'plot_' + self.arch_name + '_' + str(epoch) + '.png'
		plt.savefig(filename)

		print("Saved plots")


if __name__ == '__main__':

	a = dl_model('train')
	a.train()
	# char_to_id = {chr(97+x): x+1 for x in range(26)}
	# char_to_id['PAD'] = 0
	# a = dl_model('test_one')
	# print(a.predict("*oau", char_to_id))


