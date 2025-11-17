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
    text = str(word).lower()
    original_word = str(word)
    
    word_len = len(text)
    if word_len == 0:
        word_len = 1  
    
    vowel_chars = 'aeiou'
    consonant_chars = 'bcdfghjklmnpqrstvwxyz'
    vowels_found = sum(1 for char in text if char in vowel_chars)
    consonants_found = sum(1 for char in text if char in consonant_chars)
    
    contains_ny = 'ny' in text
    contains_ng = 'ng' in text
    has_tilde = 'ñ' in text
    
    # English-common letters (rare in Filipino)
    has_c = 'c' in text and 'ch' not in text
    has_f = 'f' in text
    has_j = 'j' in text
    has_v = 'v' in text
    has_z = 'z' in text
    has_x = 'x' in text
    
    # Common Filipino prefixes
    filipino_prefixes = ['mag', 'nag', 'pag', 'ka', 'pa', 'ma', 'na', 'um', 'in', 'maka', 'naka', 'mapa', 'napaka', 'pinaka', 'pina', 'pinag', 'pagka', 'magpa', 'nagpa', 'makapag', 'nakapag', 'mapag', 'ipag', 'ipa', 'ipinag', 'mang', 'man', 'pang', 'pan', 'pam', 'pakikipag', 'makipag', 'nakipag', 'pinaki', 'taga', 'tiga']

    starts_with_fil = any(text.startswith(prefix) for prefix in filipino_prefixes)
    
    # Common Filipino suffixes
    filipino_suffixes = ['an', 'in', 'han', 'hin', 'ng', 'on', 'yon', 'yin']
    ends_with_fil = any(text.endswith(suffix) for suffix in filipino_suffixes)
    
    # Common English prefixes 
    english_prefixes = ['un', 're', 'pre', 'dis', 'over', 'under', 'out', 'mis', 'non', 'anti', 'auto', 'inter', 'trans', 'super', 'micro', 'multi', 'semi', 'sub']
    starts_with_eng = any(text.startswith(prefix) for prefix in english_prefixes)

    # Common English suffixes 
    english_suffixes = ['tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ish', 'ive', 'ous', 'ious', 'ing', 'ed', 'er', 'est', 'ly', 'ize', 'ise', 'acy', 'ship', 'hood']
    ends_with_eng = any(text.endswith(suffix) for suffix in english_suffixes)
    
    has_repeated_chars = any(text[i] == text[i+1] for i in range(len(text)-1))
    
    starts_capital = original_word[0].isupper() if len(original_word) > 0 else False
    all_uppercase = original_word.isupper() and original_word.isalpha()
    cap_ratio = sum(1 for c in original_word if c.isupper()) / word_len
    
    has_hyphen = '-' in original_word
    
    contains_digit = any(c.isdigit() for c in original_word)
    has_special_chars = any(not c.isalnum() and c != '-' for c in original_word)
    
    vowel_ratio = vowels_found / word_len
    consonant_ratio = consonants_found / word_len
    
    filipino_bigrams = ['ng', 'ka', 'an', 'sa', 'na', 'pa', 'la', 'ta', 'ba', 'ga']
    fil_bigram_count = sum(1 for bigram in filipino_bigrams if bigram in text)
    
    english_bigrams = ['th', 'er', 'on', 'an', 'in', 'ed', 'nd', 'to', 'en', 'ty']
    eng_bigram_count = sum(1 for bigram in english_bigrams if bigram in text)
    
    repeated_vowels = ['aa', 'ee', 'ii', 'oo', 'uu']
    has_double_vowel = any(vowel_pair in text for vowel_pair in repeated_vowels)
    
    vowel_ending = text[-1] in vowel_chars if len(text) > 0 else False
    consonant_ending = text[-1] in consonant_chars if len(text) > 0 else False
    
    feature_list = [
        word_len,
        vowels_found,
        consonants_found,
        vowel_ratio,
        consonant_ratio,
        int(contains_ny),
        int(contains_ng),
        int(has_tilde),
        int(has_c),
        int(has_f),
        int(has_j),
        int(has_v),
        int(has_z),
        int(has_x),
        int(starts_with_fil),
        int(ends_with_fil),
        int(starts_with_eng),
        int(ends_with_eng),
        int(has_repeated_chars),
        int(starts_capital),
        int(all_uppercase),
        cap_ratio,
        int(has_hyphen),
        int(contains_digit),
        int(has_special_chars),
        fil_bigram_count,
        eng_bigram_count,
        int(has_double_vowel),
        int(vowel_ending),
        int(consonant_ending),
    ]
    
    return feature_list

def tag_language(tokens: List[str]) -> List[str]:
    if not tokens:
        return []

    with open('trained_model.pkl', 'rb') as model_file:
        classifier, text_vectorizer, feature_scaler = pickle.load(model_file)
    
    ngram_features = text_vectorizer.transform(tokens)
    
    numeric_features = np.array([extract_numerical_features(token) for token in tokens])
    scaled_features = feature_scaler.transform(numeric_features)
    
    combined_features = hstack([ngram_features, scaled_features])
    
    predicted_labels = classifier.predict(combined_features)
    return predicted_labels.tolist()

if __name__ == "__main__":
    test_words = ["Love", "kita", "."]
    print("Tokens:", test_words)
    result_tags = tag_language(test_words)
    print("Predicted Tags:", result_tags)