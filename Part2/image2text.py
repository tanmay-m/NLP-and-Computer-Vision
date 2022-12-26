#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Mansi Kishore Ranka, 2000933160
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

    
def train(data,train_letters,test_letters):
    total_characters = 0
    total_words = 0
    count_transition_prob = {}
    emission_probability = {}
    transition_probability = {}
    initial_probability = {}

    for i in range(len(data)):
        total_words = total_words + len(data[i])
        for j in range(len(data[i])):
            total_characters = total_characters + len(data[i][j])

            for k,d in enumerate(data[i][j]):
                if(k==0):
                    if(initial_probability.get(d)==None):
                        initial_probability[d] = 1
                    else:
                        initial_probability[d] = initial_probability[d] + 1
                
                if(k>0):
                    if(count_transition_prob.get(data[i][j][k-1])==None):
                        count_transition_prob[data[i][j][k-1]] = 1
                    else:
                        count_transition_prob[data[i][j][k-1]] = count_transition_prob[data[i][j][k-1]] + 1
                    
                    if(transition_probability.get(data[i][j][k-1])==None):
                        transition_probability[data[i][j][k-1]] = {}

                    if(transition_probability[data[i][j][k-1]].get(data[i][j][k])==None):
                        transition_probability[data[i][j][k-1]][data[i][j][k]] = 1
                    else:
                        transition_probability[data[i][j][k-1]][data[i][j][k]] = transition_probability[data[i][j][k-1]][data[i][j][k]] + 1
                else:
                    if(j!=0):
                        if(count_transition_prob.get(data[i][j-1][-1])==None):
                            count_transition_prob[data[i][j-1][-1]] = 1
                        else:
                            count_transition_prob[data[i][j-1][-1]] = count_transition_prob[data[i][j-1][-1]] + 1
                        
                        if(transition_probability.get(data[i][j-1][-1])==None):
                            transition_probability[data[i][j-1][-1]] = {}
                        
                        if(transition_probability[data[i][j-1][-1]].get(" ")==None):
                            transition_probability[data[i][j-1][-1]][" "] = 1
                        else:
                            transition_probability[data[i][j-1][-1]][" "] = transition_probability[data[i][j-1][-1]][" "] + 1


                    if(count_transition_prob.get(" ")==None):
                        count_transition_prob[" "] = 1
                    else:
                        count_transition_prob[" "] = count_transition_prob[" "] + 1

                    if(transition_probability.get(" ")==None):
                        transition_probability[" "] = {}

                    if(transition_probability[" "].get(data[i][j][k])==None):
                        transition_probability[" "][data[i][j][k]] = 1
                    else:
                        transition_probability[" "][data[i][j][k]] = transition_probability[" "][data[i][j][k]] + 1
    
    for key in initial_probability.keys():
        initial_probability[key] = initial_probability[key]/total_words
    
    for key in transition_probability.keys():
        for key_w in transition_probability[key].keys():
            transition_probability[key][key_w] = (transition_probability[key][key_w]+1)/(count_transition_prob[key]+2) ## Laplace Smoothening

    ## Calculating emission probability
    noise = 0.10
    num_test = len(test_letters)

    for i in range(num_test):
        emission_probability[i] = {}
        for letter in train_letters.keys():
            given_val = train_letters[letter]
            observed_val = test_letters[i]
            matched_val,missed_val = 0,0

            for k in range(CHARACTER_HEIGHT):
                for j in range(CHARACTER_WIDTH):
                    if(given_val[k][j]==observed_val[k][j]):
                        matched_val = matched_val + 1
                    else:
                        missed_val = missed_val + 1

            emission_probability[i][letter] = ((1-noise)**matched_val)*(noise**missed_val)
    
    return initial_probability,transition_probability,emission_probability

def simplified(test_letters,emission_probability):
    num_test = len(test_letters)
    letters = []
    for j in range(num_test):
        letters.append(max(emission_probability[j], key = emission_probability[j].get))
    
    return "".join(str(i) for i in letters)

def hmm_viterbi(test_letters,initial_probability,transition_probability,emission_probability,TRAIN_LETTERS):
    letters,key_list = [],[]
    num_test = len(test_letters)
    prev_s, curr_s = None , "s0"

    mus = {}

    for i in range(num_test):
        key_list.append(curr_s)
        mus[curr_s] = {}
        if(prev_s == None):
            for letter in TRAIN_LETTERS:
                val = 0

                if(emission_probability.get(i)!= None):
                    if(emission_probability[i].get(letter)!=None):
                        val = val + math.log(emission_probability[i][letter])
                    else:
                        val = val - 9
                else:
                    val = val - 9

                mus[curr_s][letter] = ((val + math.log(initial_probability.get(letter,0.000000001))), None)
        else:
            for letter in TRAIN_LETTERS:
                prob_letter = -sys.maxsize
                max_prob_letter = None
                for prev_letter in TRAIN_LETTERS:
                    val = 0 

                    val = val + mus[prev_s][prev_letter][0]

                    if(transition_probability.get(prev_letter)!=None):
                        if(transition_probability[prev_letter].get(letter)!=None):
                            val  = val + math.log(transition_probability[prev_letter][letter])
                        else:
                            val = val - 9
                    else:
                        val = val - 9

                    if(emission_probability.get(i)!= None):
                        if(emission_probability[i].get(letter)!=None):
                            val = val + math.log(emission_probability[i][letter])
                        else:
                            val = val - 9
                    else:
                        val = val - 9

                    if(val>prob_letter):
                        prob_letter = val
                        max_prob_letter = prev_letter
                mus[curr_s][letter] = (prob_letter,max_prob_letter)
        prev_s = curr_s
        curr_s = "s" + str(i+1)

    curr_s = key_list[-1]

    max_prob = -sys.maxsize
    max_prob_letter = None
    max_prob_parent = None

    for k in mus[curr_s].keys():
        word_prob,parent = mus[curr_s][k]

        if(word_prob>max_prob):
            max_prob = word_prob
            max_prob_letter = k
            max_prob_parent = parent
    
    letters.append(max_prob_letter)

    for i in range(len(key_list)-2,-1,-1):
        curr_s = key_list[i]

        word_prob,parent = mus[curr_s][max_prob_parent]
        max_prob_letter = max_prob_parent
        max_prob_parent = parent
        
        if(max_prob_letter == None):
                break
        letters.append(max_prob_letter)

    letters.reverse()
    return "".join(str(i) for i in letters)

               
## This code is referenced from question 1
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]
    
    lines = []
    for w in exemplars:
        words,grammer_word = w[0],w[1]
        lines.append(words)

    return lines

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    print("In load_train")
    print(len(TRAIN_LETTERS))
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
train_data = read_data(train_txt_fname)

initial_probability,transition_probability,emission_probability = train(train_data,train_letters,test_letters)

simple = simplified(test_letters, emission_probability)
print("Simple: " + simple)

hmm = hmm_viterbi(test_letters,initial_probability,transition_probability,emission_probability,TRAIN_LETTERS)
print("   HMM: " + hmm) 


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
print("\n".join([ r for r in train_letters['A'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
print("Simple: " + "Sample s1mple resu1t")
print("   HMM: " + "Sample simple result") 


