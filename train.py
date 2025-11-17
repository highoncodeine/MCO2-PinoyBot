import pandas as pd
import numpy as np
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load multiple datasets
dataset1 = pd.read_csv("CSINTSY-G28-DATASET.csv")
dataset2 = pd.read_csv("[GROUP 16] sentence_ID 780 - 832 - Sheet1.csv")
dataset3 = pd.read_csv("final_annotations1.csv")
dataset4 = pd.read_csv("final_annotations2.csv")

combined_data = pd.concat([dataset1, dataset2, dataset3, dataset4], ignore_index=True)

combined_data.columns = [col.strip() for col in combined_data.columns]
word_column = 'word'

def get_correct_label(row):
    if pd.notna(row.get('is_correct')) and row['is_correct'] == False:
        if pd.notna(row.get('corrected_label')):
            return row['corrected_label']
    return row['label']

combined_data['final_label'] = combined_data.apply(get_correct_label, axis=1)
label_column = 'final_label'

combined_data = combined_data.dropna(subset=[word_column, label_column])

if 'is_correct' in combined_data.columns:
    incorrect_entries = combined_data[combined_data['is_correct'] == False]
    correction_count = len(incorrect_entries)
    print(f"Dataset loaded: {len(combined_data)} total samples")
    print(f"Corrections applied: {correction_count} samples")
    if correction_count > 0:
        print("Sample corrections:")
        for idx in incorrect_entries.head(3).index:
            original_label = combined_data.loc[idx, 'label']
            corrected_label = combined_data.loc[idx, 'final_label']
            word_text = combined_data.loc[idx, 'word']
            print(f"  '{word_text}': {original_label} → {corrected_label}")
else:
    print("No 'is_correct' column found - using original labels")

def simplify_label(label):
    clean_label = str(label).upper().strip()
    if "ENG" in clean_label:
        return "ENG"
    elif "FIL" in clean_label:
        return "FIL"
    else:
        return "OTH"

combined_data['simplified_label'] = combined_data[label_column].apply(simplify_label)

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

text_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4), 
    max_features=300     
)
ngram_features = text_vectorizer.fit_transform(combined_data[word_column])

numerical_features = np.array([extract_numerical_features(word) for word in combined_data[word_column]])

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler(with_mean=False)  
scaled_numerical = feature_scaler.fit_transform(numerical_features)

feature_matrix = hstack([ngram_features, scaled_numerical])

target_labels = combined_data['simplified_label'].values

print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"  - N-gram features: {ngram_features.shape[1]}")
print(f"  - Numerical features: {numerical_features.shape[1]}")
print(f"  - Total features: {feature_matrix.shape[1]}")

X_train, X_temp, y_train, y_temp = train_test_split(
    feature_matrix, target_labels, test_size=0.30, random_state=42, stratify=target_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nTraining Hybrid Multinomial Naive Bayes model...")
classifier = MultinomialNB(
    alpha=0.1  
)
classifier.fit(X_train, y_train)

train_predictions = classifier.predict(X_train)
val_predictions = classifier.predict(X_val)
test_predictions = classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\n=== Accuracy Summary (Hybrid: N-grams + Numerical Features) ===")
print(f"Training Accuracy   : {train_accuracy:.4f}")
print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Test Accuracy       : {test_accuracy:.4f}")

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, test_predictions))

confusion_mat = confusion_matrix(y_test, test_predictions, labels=['ENG', 'FIL', 'OTH'])
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['ENG', 'FIL', 'OTH'],
            yticklabels=['ENG', 'FIL', 'OTH'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Hybrid Multinomial Naive Bayes (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nConfusion matrix saved as 'confusion_matrix.png'")

with open("trained_model.pkl", "wb") as model_file:
    pickle.dump((classifier, text_vectorizer, feature_scaler), model_file)

print("Hybrid Multinomial Naive Bayes model, vectorizer, and scaler saved to 'trained_model.pkl'")

print("\n" + "="*60)
print("CURRENT MODEL PERFORMANCE")
print("="*60)
print(f"Hybrid Multinomial Naive Bayes: {test_accuracy:.2%}")
print("="*60)