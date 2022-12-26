###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids: Mansi Kishore Ranka, 2000933160
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import sys
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.grammer_words_dict = {}
        self.prob_wn_sn = {}
        self.prob_wn_sprev = {}
        self.prob_sn1_sn = {}
        self.prob_sn2_sn1_sn0 = {}
        self.grammer_words = tuple(("adj","adv","adp","conj","det","noun","num","pron","prt","verb",".","x"))

    ## This part of the code is caclucated from -> https://web2.uvcs.uvic.ca/courses/elc/sample/beginner/gs/gs_55_1.htm
    ## Using this logic is inspired from -> https://github.com/gurjaspalbedi/parts-of-speech-tagging/blob/master/pos_solver.py
    ## I tried using various values from 0.8 to 0.9 and found 0.85 works the best
    def grammar_rules(self,word,tag):
        length = len(word)
        if (tag == 'verb'):
            if(word[length-3:] =="ing" or word[length-2:] == "ed" or word[length-3:] == "ify"):
                return 0.85
        if (tag == 'adj'):
            if(word[length-4:] == "like" or word[length-4:] == "less" or word[length-4:] == "able" or word[length-3:] == "ful" or word[length-3:] == "ous" or word[length-3:] == "ish" or word[length-2:] == "ic" or word[length-3:] == "ive"):
                return 0.85
        if (tag == 'adv'):
            if(word[length-2:] == "ly" ):
                return 0.85
        if (tag == 'noun'):
            if(word[length-2:] == "'s" or word[length-3:] == "ist" or word[length-3:] == "ion" or word[length-4:] == "ment"):
                return 0.85
            else:
                return 0.4
        try:
            if tag == 'num' and (word.isdigit() or (word[0]=='-' or word[0]=='+' and word[1:].isdigit())):
                return 1
        except:
            pass
        return 0.000000001

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            log_posterior = 0
            for i in range(len(sentence)):

                if(self.prob_wn_sn.get(sentence[i])!=None):
                    if(self.prob_wn_sn[sentence[i]].get(label[i])!=None):
                        log_posterior = log_posterior + math.log(self.prob_wn_sn[sentence[i]][label[i]])
                    else:
                        log_posterior = log_posterior - 9
                else:
                    log_posterior = log_posterior + math.log(self.grammar_rules(sentence[i],label[i]))

            return log_posterior
            return -999
        elif model == "HMM":
            log_posterior = 0
            for i in range(len(sentence)):

                if(self.prob_wn_sn.get(sentence[i])!=None):
                    if(self.prob_wn_sn[sentence[i]].get(label[i])!=None):
                        log_posterior = log_posterior + math.log(self.prob_wn_sn[sentence[i]][label[i]])
                    else:
                        log_posterior = log_posterior - 9
                else:
                    log_posterior = log_posterior + math.log(self.grammar_rules(sentence[i],label[i])) 

                if(i!=0):
                    log_posterior = log_posterior + math.log(self.prob_sn1_sn.get(label[i-1],0.000000001).get(label[i],0.000000001))
                else:
                    log_posterior = log_posterior + math.log(self.grammer_words_dict.get(label[i], 0.000000001))
            return log_posterior
            return -999
        elif model == "Complex":
            log_posterior = 0
            for i in range(len(sentence)):

                if(self.prob_wn_sn.get(sentence[i])!=None):
                    if(self.prob_wn_sn[sentence[i]].get(label[i])!=None):
                        log_posterior = log_posterior + math.log(self.prob_wn_sn[sentence[i]][label[i]])
                    else:
                        log_posterior = log_posterior - 9
                else:
                    log_posterior = log_posterior + math.log(self.grammar_rules(sentence[i],label[i]))

                if(i!=0):
                    log_posterior = log_posterior + math.log(self.prob_sn1_sn.get(label[i-1],0.000000001).get(label[i],0.000000001))
                    
                    if(self.prob_wn_sprev.get(sentence[i])!=None):
                        if(self.prob_wn_sprev[sentence[i]].get(label[i-1])!=None):
                            log_posterior = log_posterior + math.log(self.prob_wn_sprev[sentence[i]][label[i-1]])
                        else:
                            log_posterior = log_posterior - 9
                    else:
                        log_posterior = log_posterior + math.log(self.grammar_rules(sentence[i],label[i-1]))

                else:
                    log_posterior = log_posterior + math.log(self.grammer_words_dict.get(label[i], 0.000000001))
                if(i>=2):
                    if(self.prob_sn2_sn1_sn0.get((label[i-2],label[i-1]))==None):
                        log_posterior = log_posterior - 9
                    else:
                        log_posterior = log_posterior + math.log(self.prob_sn2_sn1_sn0.get((label[i-2],label[i-1])).get(label[i],0.000000001))

            return log_posterior
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        total_words = 0

        grammer_words_list = []
        count_sn1_sn0 = {}
        count_wn_sprev = {}

        for w in data:
            words,grammer_words = w[0],w[1]
            total_words = total_words + len(words)
            for i in range(len(words)):
                if(self.grammer_words_dict.get(grammer_words[i])==None):
                    self.grammer_words_dict[grammer_words[i]] = 1
                else:
                    self.grammer_words_dict[grammer_words[i]] = self.grammer_words_dict[grammer_words[i]]+1

                if(self.prob_wn_sn.get(words[i])==None):
                    self.prob_wn_sn[words[i]] = {}
                if(self.prob_wn_sn[words[i]].get(grammer_words[i])==None):
                    self.prob_wn_sn[words[i]][grammer_words[i]] = 1
                else:
                    self.prob_wn_sn[words[i]][grammer_words[i]] = self.prob_wn_sn[words[i]][grammer_words[i]] + 1

                if(i>=1):
                    if(self.prob_sn1_sn.get(grammer_words[i-1])==None):
                        self.prob_sn1_sn[grammer_words[i-1]] = {}
                    if(self.prob_sn1_sn[grammer_words[i-1]].get(grammer_words[i])==None):
                        self.prob_sn1_sn[grammer_words[i-1]][grammer_words[i]] = 1
                    else:
                        self.prob_sn1_sn[grammer_words[i-1]][grammer_words[i]] = self.prob_sn1_sn[grammer_words[i-1]][grammer_words[i]] + 1
                if(i>=2):
                    if(self.prob_sn2_sn1_sn0.get((grammer_words[i-2],grammer_words[i-1]))==None):
                        self.prob_sn2_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])] = {}

                    if(self.prob_sn2_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])].get(grammer_words[i])==None):
                        self.prob_sn2_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])][grammer_words[i]] = 1
                    else:
                        self.prob_sn2_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])][grammer_words[i]] = self.prob_sn2_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])][grammer_words[i]] + 1

                    if(count_sn1_sn0.get((grammer_words[i-2],grammer_words[i-1]))==None):
                        count_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])] = 1
                    else:
                        count_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])] = count_sn1_sn0[(grammer_words[i-2],grammer_words[i-1])] + 1
                    
                    if(i>=1):
                        if(count_wn_sprev.get(grammer_words[i-1])==None):
                            count_wn_sprev[grammer_words[i-1]] = 1
                        else:
                            count_wn_sprev[grammer_words[i-1]] = count_wn_sprev[grammer_words[i-1]] + 1
                        if(self.prob_wn_sprev.get(words[i])==None):
                            self.prob_wn_sprev[words[i]] = {}
                        if(self.prob_wn_sprev[words[i]].get(grammer_words[i-1])==None):
                            self.prob_wn_sprev[words[i]][grammer_words[i-1]] = 1
                        else:
                            self.prob_wn_sprev[words[i]][grammer_words[i-1]] = self.prob_wn_sprev[words[i]][grammer_words[i-1]] + 1

        for key_w in self.prob_wn_sprev.keys():
            for key in self.prob_wn_sprev[key_w].keys():
               self.prob_wn_sprev[key_w][key] =  self.prob_wn_sprev[key_w][key]/count_wn_sprev[key]

        for key_w in self.prob_wn_sn.keys():
            for key in self.prob_wn_sn[key_w].keys():
                self.prob_wn_sn[key_w][key] = self.prob_wn_sn[key_w][key]/self.grammer_words_dict[key]

        for key in self.grammer_words_dict.keys():


            for key_sn in self.prob_sn1_sn[key].keys():
                self.prob_sn1_sn[key][key_sn] = self.prob_sn1_sn[key][key_sn]/self.grammer_words_dict[key]

            self.grammer_words_dict[key] = self.grammer_words_dict[key]/total_words

        for key in self.prob_sn2_sn1_sn0.keys():
            for key_g in self.prob_sn2_sn1_sn0[key].keys():
                self.prob_sn2_sn1_sn0[key][key_g] = self.prob_sn2_sn1_sn0[key][key_g]/count_sn1_sn0[key]
        

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        path = []
        grammer_list = self.grammer_words

        for s in sentence:
            prob_word = -sys.maxsize
            max_prob_pos = None
            for g in grammer_list:
            
                val = 0


                if(self.prob_wn_sn.get(s)!=None):
                    if(self.prob_wn_sn[s].get(g)!=None):
                        val = val + math.log(self.prob_wn_sn[s][g])
                    else:
                        val = val + math.log(0.000000001)
                else:
                    val = val + math.log(0.000000001)
            
                if(prob_word<val):
                    prob_word = val
                    max_prob_pos = g

            if(max_prob_pos!=None):
                path.append(max_prob_pos)
            else:
                path.append("x")
        return path
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):

        path, key_list = [],[]
        total_keys = 0
        grammer_list = self.grammer_words
        prev_s, curr_s = None , "s0"

        mus = {}

        for i,s in enumerate(sentence):
            prob_word = -sys.maxsize
            max_prob_pos = "x"
            mus[curr_s] = {}
            if(prev_s==None):
                prob_word = -sys.maxsize
                max_prob_pos = "x"
                for curr_s_g in grammer_list:
                    val = 0

                    if(self.prob_wn_sn.get(s)!=None):
                        if(self.prob_wn_sn[s].get(curr_s_g)!=None):
                            val = val + math.log(self.prob_wn_sn[s][curr_s_g])
                        else:
                            val = val + math.log(0.000000001)
                    else:
                        val = val +  math.log(self.grammar_rules(s,curr_s_g))
                    mus[curr_s][curr_s_g] = (val + math.log(self.grammer_words_dict.get(curr_s_g,0.000000001)),None)
                key_list.append(curr_s)
                total_keys = total_keys + 1
            else:
                for curr_s_g in grammer_list:
                    prob_word = -sys.maxsize
                    max_prob_pos = "x"
                    for prev_s_g in grammer_list:
                        val = 0

                        if(self.prob_wn_sn.get(s)!=None):
                            if(self.prob_wn_sn[s].get(curr_s_g)!=None):
                                val = val + math.log(self.prob_wn_sn[s][curr_s_g])
                            else:
                                val = val + math.log(0.000000001)
                        else:
                            val = val + math.log(self.grammar_rules(s,curr_s_g))

                        if(self.prob_sn1_sn.get(prev_s_g)!=None):
                            if(self.prob_sn1_sn[prev_s_g].get(curr_s_g)!=None):
                                val = val + math.log(self.prob_sn1_sn[prev_s_g][curr_s_g])
                            else:
                                val = val + math.log(0.000000001)
                        else:
                            val = val + math.log(0.000000001)
                        val = mus[prev_s][prev_s_g][0] + val
                        if(val>prob_word):
                            prob_word = val
                            max_prob_pos = prev_s_g
                    mus[curr_s][curr_s_g] = (prob_word,max_prob_pos) 
                key_list.append(curr_s)
                total_keys = total_keys + 1

            prev_s = curr_s
            curr_s = "s"+ str(i+1)
        

        key_list.reverse()

        curr_s = key_list[0]
        max_prob = -sys.maxsize
        max_prob_pos = None
        max_prob_parent = None

        for k in mus[curr_s].keys():
            word_prob,parent = mus[curr_s][k]

            if(word_prob>max_prob):
                max_prob = word_prob
                max_prob_pos = k
                max_prob_parent = parent

        path.append(max_prob_pos)


        for i in range(1,len(key_list)):
            curr_s = key_list[i]
            word_prob,parent = mus[curr_s][max_prob_parent]

            max_prob_pos = max_prob_parent
            max_prob_parent = parent

            if(max_prob_pos == None):
                break
            path.append(max_prob_pos)


        path.reverse()

        return path
        return [ "noun" ] * len(sentence)

    ## This function is inspired from -> https://github.com/gurjaspalbedi/parts-of-speech-tagging/blob/master/pos_solver.py
    ## This function is inspired from the generate_sample function of the above link
    def generate_label(self,sentence,label):

        for i in range(len(sentence)):
            
            log_probs = np.zeros(len(self.grammer_words))
            for j,grammer_word in enumerate(self.grammer_words):
                label[i] = grammer_word
                log_probs[j] =  self.posterior("Complex",sentence,label)
            min_log_prob = np.min(log_probs)
            
            prob_array = np.zeros(len(self.grammer_words))
            for k in range(len(log_probs)):
                log_probs[k] = log_probs[k] - min_log_prob
                prob_array[k] = math.exp(log_probs[k]) + 0.0000000001 ## Adding a small number to avoid division by 0
            
            prob_array = prob_array/np.sum(prob_array)

            label[i] = self.grammer_words[(int)(np.random.choice(np.arange(0,len(self.grammer_words)), 1, p = prob_array))]

        return label

    ## This part of code is inspired from -> https://github.com/gurjaspalbedi/parts-of-speech-tagging/blob/master/pos_solver.py
    def complex_mcmc(self, sentence):


        ## Initializing the sample using viterbi -> This is time consuming and we only need an estimate so I will use simplified assumption
        #label = self.hmm_viterbi(sentence)
        label = self.simplified(sentence)

        num_iter = 150
        burning_iter = 20
        labels = []

        for itr in range(num_iter):
            label = self.generate_label(sentence,label)
            #if(itr>burning_iter):
            labels.append(label)
        
        count_labels = {}


        final_label = []

        for i in range(len(sentence)):
            count_labels = {}
            for j in range(len(labels)):
                label = labels[j]
                if(count_labels.get(label[i])==None):
                    count_labels[label[i]] = 1
                else:
                    count_labels[label[i]] = count_labels[label[i]] + 1

            final_label.append(max(count_labels, key = count_labels.get))
            

        return final_label

        


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

