#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nltk

from nltk.tokenize import RegexpTokenizer
import syllables_en # file ezafe shod

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
SPECIAL_CHARS = ['.', ',', '!', '?']


def get_words(text=''):
    words = []
    words = TOKENIZER.tokenize(text)
    filtered_words = []
    for word in words:
        if word in SPECIAL_CHARS or word == " ":
            pass
        else:
            new_word = word.replace(",","").replace(".","")
            new_word = new_word.replace("!","").replace("?","")
            filtered_words.append(new_word)
    return filtered_words

def get_sentences(text=''):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # sentences = tokenizer.tokenize(text)
    sentences = tokenizer.tokenize(text.decode('utf-8'))
    return sentences

def count_syllables(words):
    syllableCount = 0
    for word in words:
        syllableCount += syllables_en.count(word)
    return syllableCount


def analyze_text(self, text):
        words = get_words(text)
        word_count = len(words)
        sentence_count = len(get_sentences(text))
        syllable_count = count_syllables(words)
        avg_words_p_sentence = word_count/sentence_count
        
        self.analyzedVars = {
            'word_cnt': float(word_count),
            'syllable_cnt': float(syllable_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
        }

def FleschKincaidGradeLevel(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.39 * (self.analyzedVars['avg_words_p_sentence']) + 11.8 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']) - 15.59
        return round(score, 4)

#from readability import Readability
#import sys
#
#if __name__ == '__main__':
#    infile = sys.argv[1]
#    text = open(infile).read()
#    rd = Readability(text)
#    print(rd.FleschKincaidGradeLevel())
#
#
