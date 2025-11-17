"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project,
which identifies the language of each word in a code-switched Filipino-English text.
The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").
"""

import pickle
import string
import numpy as np
from scipy.sparse import hstack
from typing import List

def extract_numerical_features(word):
    word_lower = str(word).lower()
    word_orig = str(word)
    
    length = len(word_lower)
    if length == 0:
        length = 1  
    
    vowels = 'aeiou'
    consonants = 'bcdfghjklmnpqrstvwxyz'
    vowel_count = sum(1 for c in word_lower if c in vowels)
    consonant_count = sum(1 for c in word_lower if c in consonants)
    
    has_ny = 'ny' in word_lower
    has_ng = 'ng' in word_lower
    has_tilde = 'ñ' in word_lower
    
    # English-common letters (rare in Filipino)
    has_c = 'c' in word_lower and 'ch' not in word_lower
    has_f = 'f' in word_lower
    has_j = 'j' in word_lower
    has_v = 'v' in word_lower
    has_z = 'z' in word_lower
    has_x = 'x' in word_lower
    
    # Common Filipino prefixes
    fil_prefixes = ['mag', 'nag', 'pag', 'ka', 'pa', 'ma', 'na', 'um', 'in', 'maka', 'naka', 'mapa', 'napaka', 'pinaka', 'pina', 'pinag', 'pagka', 'magpa', 'nagpa', 'makapag', 'nakapag', 'mapag', 'ipag', 'ipa', 'ipinag', 'mang', 'man', 'pang', 'pan', 'pam', 'pakikipag', 'makipag', 'nakipag', 'pinaki', 'taga', 'tiga']

    has_fil_prefix = any(word_lower.startswith(p) for p in fil_prefixes)
    
    # Common Filipino suffixes
    fil_suffixes = ['an', 'in', 'han', 'hin', 'ng', 'on', 'yon', 'yin']
    has_fil_suffix = any(word_lower.endswith(s) for s in fil_suffixes)
    
    # Common English prefixes 
    eng_prefixes = ['un', 're', 'pre', 'dis', 'over', 'under', 'out', 'mis', 'non', 'anti', 'auto', 'inter', 'trans', 'super', 'micro', 'multi', 'semi', 'sub']
    has_eng_prefix = any(word_lower.startswith(p) for p in eng_prefixes)

    # Common English suffixes 
    eng_suffixes = ['tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ish', 'ive', 'ous', 'ious', 'ing', 'ed', 'er', 'est', 'ly', 'ize', 'ise', 'acy', 'ship', 'hood']
    has_eng_suffix = any(word_lower.endswith(s) for s in eng_suffixes)
    
    has_repeated_chars = any(word_lower[i] == word_lower[i+1] for i in range(len(word_lower)-1))
    
    is_capitalized = word_orig[0].isupper() if len(word_orig) > 0 else False
    is_all_caps = word_orig.isupper() and word_orig.isalpha()
    capital_ratio = sum(1 for c in word_orig if c.isupper()) / length
    
    has_hyphen = '-' in word_orig
    
    has_digit = any(c.isdigit() for c in word_orig)
    has_special = any(not c.isalnum() and c != '-' for c in word_orig)
    
    vowel_ratio = vowel_count / length
    consonant_ratio = consonant_count / length
    
    fil_bigrams = ['ng', 'ka', 'an', 'sa', 'na', 'pa', 'la', 'ta', 'ba', 'ga']
    fil_bigram_count = sum(1 for bg in fil_bigrams if bg in word_lower)
    
    eng_bigrams = ['th', 'er', 'on', 'an', 'in', 'ed', 'nd', 'to', 'en', 'ty']
    eng_bigram_count = sum(1 for bg in eng_bigrams if bg in word_lower)
    
    double_vowels = ['aa', 'ee', 'ii', 'oo', 'uu']
    has_double_vowel = any(dv in word_lower for dv in double_vowels)
    
    ends_with_vowel = word_lower[-1] in vowels if len(word_lower) > 0 else False
    ends_with_consonant = word_lower[-1] in consonants if len(word_lower) > 0 else False
    
    features = [
        length,
        vowel_count,
        consonant_count,
        vowel_ratio,
        consonant_ratio,
        int(has_ny),
        int(has_ng),
        int(has_tilde),
        int(has_c),
        int(has_f),
        int(has_j),
        int(has_v),
        int(has_z),
        int(has_x),
        int(has_fil_prefix),
        int(has_fil_suffix),
        int(has_eng_prefix),
        int(has_eng_suffix),
        int(has_repeated_chars),
        int(is_capitalized),
        int(is_all_caps),
        capital_ratio,
        int(has_hyphen),
        int(has_digit),
        int(has_special),
        fil_bigram_count,
        eng_bigram_count,
        int(has_double_vowel),
        int(ends_with_vowel),
        int(ends_with_consonant),
    ]
    
    return features

def tag_language(tokens: List[str]) -> List[str]:
    if not tokens:
        return []

    with open('trained_model.pkl', 'rb') as f:
        model, vectorizer, scaler = pickle.load(f)
    
    X_ngrams = vectorizer.transform(tokens)
    
    X_numerical = np.array([extract_numerical_features(token) for token in tokens])
    X_numerical_scaled = scaler.transform(X_numerical)
    
    X_combined = hstack([X_ngrams, X_numerical_scaled])
    
    predictions = model.predict(X_combined)
    return predictions.tolist()

if __name__ == "__main__":
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Predicted Tags:", tags)