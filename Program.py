import codecs
import unicodedata
import random
import math
from collections import defaultdict, Counter

import numpy as np
import pylab as plt
from scipy.stats import gaussian_kde
import time

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
PRINTABLE = set(('Lu', 'Ll', 'Zs'))
WINDOWS_SIZE = 79

def random_key():
    '''
    Returns a random permutation of the alphabet
    '''
    random_key = list(ALPHABET)
    random.shuffle(random_key)
    return random_key

def simplify(s):
    '''
    Function to remove any UTF-8 character transforming it to a [A-Z]
    character, any other symbol or punctuation is transformed into a 
    space.
    '''
    return ' '.join(''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) in PRINTABLE).split()).upper()

def count_unigram_frequency(text):
    '''
    Returns a dictionary with the frequency of each letter in the text, 
    for simplicity all the letters have at least 1 occurrence. 
    '''
    # to avoid problems with 0 letters
    return Counter(text + ''.join(ALPHABET))

def count_bigram_frequency(text):
    '''
    Returns a dictionary with the frequency of each bigram in the text, 
    for simplicity all possible bigrams have at least 1 occurrence. 
    '''
    # to avoid problems with 0 letters
    all = [chr1 + chr2 for chr1 in ALPHABET for chr2 in ALPHABET]
    return Counter([text[i-1:i+1] for i in range(1, len(text))] + all)

def read_and_simply_text(filename):
    '''
    Read all lines file and removes all special characters, 
    at the end the text only have [A-Z] characters plus the space
    '''
    file = codecs.open(filename,'r','utf-8')
    return simplify(' '.join(file.readlines()))

################################################################################
# Unigram substitution code
################################################################################

def unigram_frequency_solver(train_text, encrypted_text):
    '''
    Simple transformation by unigram occurrence (1:1 mapping by 
    the frequencies in the train text and the ciphered text.
    '''
    train_text_uni_freq = count_unigram_frequency(train_text)
    enc_text_uni_freq = count_unigram_frequency(encrypted_text)

    keys = defaultdict()
    # dec and train are key:value pairs
    for dec, train in zip(enc_text_uni_freq.most_common(), train_text_uni_freq.most_common()):
        keys[dec[0]] = train[0] # using only the key
    # unigram substitution, return decrypted text, key
    return ''.join([keys[chr] for chr in encrypted_text]), [keys[chr] for chr in ALPHABET]

################################################################################
# Bigram substitution code
################################################################################

def encrypt_by_key_substitution(plain_text, key):
    '''
    Returns the text encrypted using the key
    '''
    dct = dict(zip(ALPHABET, key))
    return ''.join([dct[chr] for chr in plain_text])

def decrypt_by_key(encrypted_text, key):
    '''
    Returns the encrypted text decrypted up to the "length" position.
    '''
    dct = dict(zip(ALPHABET, key))
    return ''.join([dct[chr] for chr in encrypted_text])

def bigram_log_score(train_counter, encrypted_counter, key):
    '''
    Return the Pi function in the Metropolis algorithm (taking the log to de 
    equation (2) in the paper.
    '''
    # mapping between the alphabet and the encription key
    pairs = [chr1 + chr2 for chr1 in ALPHABET for chr2 in ALPHABET]
    dct = dict(zip(pairs, [chr1 + chr2 for chr1 in key for chr2 in key]))
    #
    return sum([math.log(train_counter[dct[pair]]) * encrypted_counter[pair] for pair in pairs])

def bigram_frequency_solver(train_text, encrypted_text, key = random_key(), iterations = 10000):
    # initial settings for the key
    train_counter = count_bigram_frequency(train_text)
    encrypted_counter = count_bigram_frequency(encrypted_text)
    # initial Pi(x_0) in the metropolis algorithm
    score = bigram_log_score(train_counter, encrypted_counter, key)

    # power is a exponentiation bias in the rejection of samples
    power = 1
    
    key_sequences = []
    max_key = None
    max_score = 0
    for it in range(iterations):
        new_key = list(key)
        pos1, pos2 = random.randint(0, len(key)-1), random.randint(0, len(key)-1)
        new_key[pos1], new_key[pos2] = new_key[pos2], new_key[pos1]
        # X_{i+1}
        new_score = bigram_log_score(train_counter, encrypted_counter, new_key)

        # metropolis rejection step
        if math.log(random.random()) < power * (new_score - score):
            key = new_key
            score = new_score
            # saving the maximum
            if score > max_score:
                max_key = key
                max_score = score
                print(decrypt_by_key(encrypted_text[:WINDOWS_SIZE], key) + '\r', end='')
                time.sleep(0.2)
                key_sequences.append(key)
    
    print(decrypt_by_key(encrypted_text[:WINDOWS_SIZE], key))
    return max_score, max_key, key_sequences

################################################################################
# Tests
################################################################################

def substitution_test():
    test_len = 2000
    mcmc_simulations = 3
    mcmc_iterations = 15000
    encryption_language_idx = 0
    test_filenames = ['olivertwist.txt', 'tradicionesperuanas.txt', 'lecomtedemontecristo.txt', 'ladivinacommediadidante.txt']
    train_filenames = ['warandpeace.txt', 'donquijote.txt', 'lesmiserables.txt', 'decameron.txt']    
    languages = ['english', 'spanish', 'french', 'italian']
    encrypted_key = random_key()

    best_language_score = 0.0
    best_lang_idx = 0
    best_lang_sequence = None

    complete_encrypted_text = read_and_simply_text(test_filenames[encryption_language_idx])
    complete_encrypted_text = encrypt_by_key_substitution(complete_encrypted_text, encrypted_key)

    print('Language:', languages[encryption_language_idx])
    print('Text to be encrypted:', test_filenames[encryption_language_idx])
    print('Key used:', ''.join(encrypted_key))
    print()
    print('===================================================================')
    print('Running Metropolis-Hastings')
    print()

    rand_start = [random.randint(0, len(complete_encrypted_text) - test_len) for i in range(mcmc_simulations)]
    for lang_idx in range(len(languages)):
        print('Language:', languages[lang_idx])
        train_filename = train_filenames[lang_idx]
        train_text = read_and_simply_text(train_filename)
        train_word_set = set(train_text.split())
        
        ## according to the text the unigram solver helps, I think that's true
        ## but only if the encrypted text is statistical representative of the 
        ## whole, for example in the second text, dosen't helps very much.
        max_score = 0
        max_key = None
        max_sequence = None
        for i in range(mcmc_simulations):
                
            encrypted_text = complete_encrypted_text[rand_start[i] : rand_start[i] + test_len]
        
            unigram_text, key = unigram_frequency_solver(train_text, encrypted_text)
            #key = random_key()
            score, key, sequence = bigram_frequency_solver(train_text, encrypted_text, key, mcmc_iterations)
            if score > max_score:
                max_score = score
                max_key = key
                max_sequence = sequence

        decrypted_words = decrypt_by_key(encrypted_text, max_key).split()
        language_score = sum([1 for word in decrypted_words if word in train_word_set])/len(decrypted_words)
        print()
        print('Language score:', language_score)
        dct = dict(zip(max_key, ALPHABET))
        print('Key Found:',''.join([dct[chr] for chr in ALPHABET]))
        print('Real Key :',''.join(encrypted_key))
        print()

        if language_score > best_language_score:
            best_lang_idx = lang_idx
            best_lang_sequence = max_sequence
            best_language_score = language_score
                
    generate_animation(train_filenames[best_lang_idx], test_filenames[best_lang_idx], encrypted_key, best_lang_sequence)
    print('Most likely language:', languages[best_lang_idx])

def generate_animation(train_filename, test_filename, encrypted_key, sequence):
    pairs = [chr1 + chr2 for chr1 in ALPHABET for chr2 in ALPHABET]
    freq1 = count_bigram_frequency(read_and_simply_text(train_filename))
    freq2 = count_bigram_frequency(encrypt_by_key_substitution(read_and_simply_text(test_filename), encrypted_key))
    # desired key
    dct = dict(zip(pairs, [chr1 + chr2 for chr1 in encrypted_key for chr2 in encrypted_key]))
    data = [math.log(freq1[pair]*freq2[dct[pair]]) for pair in pairs]
    density = gaussian_kde(data)
    xs = np.linspace(0,8,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    final = density(xs)
    line1, = ax.plot(xs, final, 'b')
    line2, = ax.plot(xs, final, 'r')

    for key in sequence:
        # inverting the key
        dct = dict(zip(key, ALPHABET))
        key = [dct[chr] for chr in ALPHABET]
        # finding similarity between key
        dct = dict(zip(pairs, [chr1 + chr2 for chr1 in key for chr2 in key]))
        data1 = [math.log(freq1[pair]*freq2[dct[pair]]) for pair in pairs]
        density = gaussian_kde(data1)
        xs = np.linspace(0,8,200)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        line1.set_ydata(density(xs))
        fig.canvas.draw()
        time.sleep(0.5)
    #plt.show()

if __name__ == '__main__':
    substitution_test()    
    input("Click Enter to Finish...")