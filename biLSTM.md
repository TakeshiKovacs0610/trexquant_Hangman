Directory structure:
└── using bi-LSTM/
    ├── README.md
    ├── hangman_api_user.ipynb
    ├── main.py
    ├── train.py
    └── utils.py


Files Content:

================================================
FILE: using bi-LSTM/README.md
================================================
# trexquantHangmanChallenge
Hangman challenge by trexquant is a challenge to predict words, letter by letter.

## Vowel Prior Probability
Vowels: [a, e, i, o, u]
Created dictionary vowel_prior such that
vowel_prior = {}
keys: length of words
values: the probability of vowels given the length of words

## Data Encoding
### Input Data
Permutation:
From ~220k words dictionary, we have created around 10 million words by masking different letters in the word, i.e., by replacing letters with underscore.

The maximum length of a word in the given dictionary is 29. Testing will happen on mutually exclusive datasets. Thus max word length is assumed at 35.
Each input word is encoded to a 35-dimensional vector, such that alphabets {a-z} will be replaced by numbers {1-26} and underscore will be replaced by 27. The vector will be pre-padded.
Thus, the masked word "aa__" of the word "aaa" will be encoded as 
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,27]

### Target Data
For each of the masked input words, the output will be the original unmasked word. This word has been encoded into a 26-dimensional vector with each position representing letters of the alphabet from a to z.
Thus the output encoding for the word "aaa" will be:
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

As the word contains only one letter "a", output encoding will have 1 at the first position.

## Modelling
Encoding + bi-LSTM has been built to train on this data.

## Prediction Strategy
It is required to predict the word within 6 incorrect tries.

1. Vowel Prediction:
   Leveraging Vowel_priors, we will guess the top vowel if
     tries_remains > 4 and len(guessed_letters) <= max_vowel_guess_limit
2. The remaining tries will be utilized by the bi-lstm model
3. The prediction will happen letter-by-letter



================================================
FILE: using bi-LSTM/hangman_api_user.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Trexquant Interview Project (The Hangman Game)

* Copyright Trexquant Investment LP. All Rights Reserved. 
* Redistribution of this question without written consent from Trexquant is prohibited
"""

"""
## Instruction:
For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server. 

When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
or (2) the user has made six incorrect guesses.

You are required to write a "guess" function that takes current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games. You have the opportunity to practice before you want to start recording your game results.

Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.

You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.

This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.
"""

import json
import secrets
import time
import re
import collections
import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn



def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()                
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            loss_estimate.append(loss)
            batch_no.append(current)
            epoch_no.append(epoch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in data_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        sample = {"features": features, "label": label}
        return features, label

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.Sequential(
            nn.Embedding(64, 32, max_norm=1, norm_type=2),
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True),
            extract_tensor(),
            nn.Linear(128, 26)
        )
    
    def forward(self, x):
        logits = self.LSTM_stack(x)
        return logits

def create_dataloader(input_tensor, target_tensor):
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=128, shuffle=True)
    return all_features_dataloader

def save_model(model):
    torch.save(model.state_dict(), "bi-lstm-embedding-model-state.pt")    

def train_model(input_tensor, target_tensor):
    all_features_dataloader = create_dataloader(input_tensor, target_tensor)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_estimate = []
    batch_no = []
    epoch_no = []
    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(all_features_dataloader, model, loss_fn, optimizer, loss_estimate, batch_no, t, epoch_no)
        test_loop(all_features_dataloader, model, loss_fn)
    print("Done!")
    save_model(model)

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)

"""
# API Usage Examples
"""

"""
## To start a new game:
1. Make sure you have implemented your own "guess" method.
2. Use the access_token that we sent you to create your HangmanAPI object. 
3. Start a game by calling "start_game" method.
4. If you wish to test your function without being recorded, set "practice" parameter to 1.
5. Note: You have a rate limit of 20 new games per minute. DO NOT start more than 20 new games within one minute.
"""

# api = HangmanAPI(access_token="INSERT_YOUR_TOKEN_HERE", timeout=2000)


"""
## Playing practice games:
You can use the command below to play up to 100,000 practice games.
"""

"""
api.start_game(practice=1,verbose=True)
[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
practice_success_rate = total_practice_successes / total_practice_runs
print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))
"""
# Output:
#   "\napi.start_game(practice=1,verbose=True)\n[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)\npractice_success_rate = total_practice_successes / total_practice_runs\nprint('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))\n"

"""
## Playing recorded games:
Please finalize your code prior to running the cell below. Once this code executes once successfully your submission will be finalized. Our system will not allow you to rerun any additional games.

Please note that it is expected that after you successfully run this block of code that subsequent runs will result in the error message "Your account has been deactivated".

Once you've run this section of the code your submission is complete. Please send us your source code via email.
"""

"""
for i in range(1000):
    print('Playing ', i, ' th game')
    # Uncomment the following line to execute your final runs. Do not do this until you are satisfied with your submission
    #api.start_game(practice=0,verbose=False)
    
    # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
    time.sleep(0.5)
"""
# Output:
#   "\nfor i in range(1000):\n    print('Playing ', i, ' th game')\n    # Uncomment the following line to execute your final runs. Do not do this until you are satisfied with your submission\n    #api.start_game(practice=0,verbose=False)\n    \n    # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests\n    time.sleep(0.5)\n"

"""
## To check your game statistics
1. Simply use "my_status" method.
2. Returns your total number of games, and number of wins.
"""

"""
[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
success_rate = total_recorded_successes/total_recorded_runs
print('overall success rate = %.3f' % success_rate)
"""
# Output:
#   "\n[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)\nsuccess_rate = total_recorded_successes/total_recorded_runs\nprint('overall success rate = %.3f' % success_rate)\n"



================================================
FILE: using bi-LSTM/main.py
================================================
import pandas as pd
import numpy as np
from utils import *
import train

if __name__ == '__main__':
    input_tensor, target_tensor = get_datasets()
    train_model(input_tensor, target_tensor)



================================================
FILE: using bi-LSTM/train.py
================================================
import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

def train_loop(data_loader, model, loss_fn, optimizer, loss_estimate, batch_no, epoch, epoch_no):
    size = len(data_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()                
        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            loss_estimate.append(loss)
            batch_no.append(current)
            epoch_no.append(epoch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    model.eval()
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (X, y) in data_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDatasetTrain(Dataset):
    def __init__(self, X_train, y_train):
        self.features = X_train
        self.label = y_train
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.label[idx]
        sample = {"features": features, "label": label}
        return features, label

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_stack = nn.Sequential(
            nn.Embedding(64, 32, max_norm=1, norm_type=2),
            nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True),
            extract_tensor(),
            nn.Linear(128, 26)
        )
    
    def forward(self, x):
        logits = self.LSTM_stack(x)
        return logits

def create_dataloader(input_tensor, target_tensor):
    all_features_data = CustomDatasetTrain(input_tensor, target_tensor)
    all_features_dataloader = DataLoader(all_features_data, batch_size=128, shuffle=True)
    return all_features_dataloader

def save_model(model):
    torch.save(model.state_dict(), "bi-lstm-embedding-model-state.pt")    

def train_model(input_tensor, target_tensor):
    all_features_dataloader = create_dataloader(input_tensor, target_tensor)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_estimate = []
    batch_no = []
    epoch_no = []
    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(all_features_dataloader, model, loss_fn, optimizer, loss_estimate, batch_no, t, epoch_no)
        test_loop(all_features_dataloader, model, loss_fn)
    print("Done!")
    save_model(model)



================================================
FILE: using bi-LSTM/utils.py
================================================
import pandas as pd
import numpy as np
import os
import random
import string
import pickle
import torch


def get_char_mapping():
    char_mapping = {'_': 27}
    ct = 1
    for i in list(string.ascii_lowercase):
        char_mapping[i] = ct
        ct = ct + 1
    return char_mapping

def create_intermediate_data(df):
    x = pd.DataFrame(df.split('\n'))
    x[1] = x[0].apply(lambda p: len(p))
    x['vowels_present'] = x[0].apply(lambda p: set(p).intersection({'a', 'e', 'i', 'o', 'u'}))
    x['vowels_count'] = x['vowels_present'].apply(lambda p: len(p))
    x['unique_char_count'] = x[0].apply(lambda p: len(set(p)))
    x_ = x[~((x['unique_char_count'].isin([0, 1, 2])) | (x[1] <= 3)) & (x.vowels_count != 0)]

def read_data():
    with open("./Data/words_250000_train.txt", "r") as f:
        df = f.read()
    return df

def loop_for_permutation(unique_letters, word, all_perm, i):
    random_letters = random.sample(unique_letters, i+1)
    new_permuted_word = word
    for letter in random_letters:
        new_permuted_word = new_permuted_word.replace(letter, "_")
        all_perm.append(new_permuted_word)

def permute_all(word, vowel_permutation_loop=False):
    unique_letters = list(set(word))
    all_perm = []
    if vowel_permutation_loop:
        for i in range(len(unique_letters) - 1):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm
    else:
        for i in range(len(unique_letters) - 2):
            loop_for_permutation(unique_letters, word, all_perm, i)
        all_perm = list(set(all_perm))
        return all_perm

def permute_consonents(word):
    len_word = len(word)
    vowel_word = "".join([i if i in ["a", "e", "i", "o", "u"] else "_" for i in list(word)])
    vowel_idxs = []
    for i in range(len(vowel_word)):
        if vowel_word[i] == "_":
            continue
        else:
            vowel_idxs.append(i)  
    abridged_vowel_word = vowel_word.replace("_", "")
    all_permute_consonents = permute_all(abridged_vowel_word, vowel_permutation_loop=True)
    permuted_consonents = []
    for permuted_word in all_permute_consonents:
        a = ["_"] * len(word)
        vowel_no = 0
        for vowel in permuted_word:
            a[vowel_idxs[vowel_no]] = vowel
            vowel_no += 1
        permuted_consonents.append("".join(a))
    return permuted_consonents

def create_masked_dictionary(df_aug):
    masked_dictionary = {}
    counter = 0
    for word in df_aug[0]:
        all_masked_words_for_word = []
        all_masked_words_for_word = all_masked_words_for_word + permute_all(word)
        all_masked_words_for_word = all_masked_words_for_word +  permute_consonents(word)
        all_masked_words_for_word = list(set(all_masked_words_for_word))
        masked_dictionary[word] = all_masked_words_for_word
        if counter % 10000 == 0:
            print(f"Iteration {counter} completed")
        counter = counter + 1

def get_vowel_prob(df_vowel, vowel):
    try:
        return df_vowel[0].apply(lambda p: vowel in p).value_counts(normalize=True).loc[True]
    except:
        return 0

def get_vowel_prior(df_aug):
    prior_json = {}
    for word_len in range(df_aug[1].max()):
        prior_json[word_len + 1] = []
        df_vowel = df_aug[df_aug[1] == word_len]
        for vowel in ['a', 'e', 'i', 'o', 'u']:
            prior_json[word_len + 1].append(get_vowel_prob(df_vowel, vowel))
        prior_json[word_len + 1] = pd.DataFrame([pd.Series(['a', 'e', 'i', 'o', 'u']), pd.Series(prior_json[word_len + 1])]).T.sort_values(by=1, ascending=False)
    return prior_json    

def save_vowel_prior(vowel_prior):
    pickle.dump(vowel_prior, open("prior_probabilities.pkl", "wb"))    

def encode_output(word):
    char_mapping = get_char_mapping()
    output_vector = [0] * 26
    for letter in word:
        output_vector[char_mapping[letter] - 1] = 1
#     return torch.tensor([output_vector])
    return output_vector

def encode_input(word):
    char_mapping = get_char_mapping()
    given_word_len = len(word)
    embedding_len = 35
    word_vector = [0] * embedding_len
    ct = 0
    for letter_no in range(embedding_len - given_word_len, embedding_len):
        word_vector[letter_no] = char_mapping[word[ct]]
        ct += 1
    return word_vector

def encode_words(masked_dictionary): 
    target_data = []
    input_data = []
    counter = 0
    for output_word, input_words in masked_dictionary.items():
        output_vector = encode_output(output_word)
        for input_word in input_words:
            target_data.append(output_vector)
            input_data.append(encode_input(input_word))
        if counter % 10000 == 0:
            print(f"Iteration {counter} completed")
        counter += 1
    return input_data, target_data

def save_input_output_data(input_data, target_data):
    with open(r'input_features.txt', 'w') as fp:
        for item in input_data:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    with open(r'target_features.txt', 'w') as fp:
        for item in target_data:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

def convert_to_tensor(input_data, target_data):
    input_tensor = torch.tensor(input_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)
    return input_tensor, target_tensor

def get_datasets():
    df = read_data()
    x_ = create_intermediate_data(df)
    df_aug = x_.copy()
    masked_dictionary = create_masked_dictionary(df_aug)
    vowel_prior = get_vowel_prior(df_aug)
    save_vowel_prior(vowel_prior)
    input_data, target_data = encode_words(masked_dictionary)
    save_input_output_data(input_data, target_data)
    input_tensor, target_tensor = convert_to_tensor(input_data, target_data)
    





