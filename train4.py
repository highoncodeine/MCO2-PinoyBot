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
data1 = pd.read_csv("CSINTSY-G28-DATASET.csv")

data = pd.concat([data1], ignore_index=True)

data.columns = [col.strip() for col in data.columns]
token_col = 'word'

def get_correct_label(row):
    if pd.notna(row.get('is_correct')) and row['is_correct'] == False:
        if pd.notna(row.get('corrected_label')):
            return row['corrected_label']
    return row['label']

data['final_label'] = data.apply(get_correct_label, axis=1)
label_col = 'final_label'

data = data.dropna(subset=[token_col, label_col])

if 'is_correct' in data.columns:
    incorrect_rows = data[data['is_correct'] == False]
    total_corrections = len(incorrect_rows)
    print(f"Dataset loaded: {len(data)} total samples")
    print(f"Corrections applied: {total_corrections} samples")
    if total_corrections > 0:
        print("Sample corrections:")
        for idx in incorrect_rows.head(3).index:
            original = data.loc[idx, 'label']
            corrected = data.loc[idx, 'final_label']
            word = data.loc[idx, 'word']
            print(f"  '{word}': {original} → {corrected}")
else:
    print("No 'is_correct' column found - using original labels")

def simplify_label(label):
    label = str(label).upper().strip()
    if "ENG" in label:
        return "ENG"
    elif "FIL" in label:
        return "FIL"
    else:
        return "OTH"

data['label_simplified'] = data[label_col].apply(simplify_label)

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
    eng_prefixes = ['a', 'an', 'ante', 'anti', 'auto', 'circum', 'co', 'com', 'con', 'contra', 'contro', 'de', 'dis', 'en', 'ex', 'extra', 'hetero', 'homo', 'homeo', 'hyper', 'il', 'im', 'in', 'ir', 'in', 'inter', 'intra', 'intro', 'macro', 'micro', 'mono', 'non', 'omni', 'post', 'pre', 'pro', 're', 'sub', 'sym', 'syn', 'tele', 'trans', 'tri', 'un', 'uni', 'up']
    has_eng_prefix = any(word_lower.startswith(p) for p in eng_prefixes)

    # Common English suffixes
    eng_suffixes = ['acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion', 'ate', 'en', 'ify', 'fy', 'ize', 'ise', 'able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish', 'ive', 'less', 'y']
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

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4), 
    max_features=300     
)
X_ngrams = vectorizer.fit_transform(data[token_col])

X_numerical = np.array([extract_numerical_features(word) for word in data[token_col]])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)  
X_numerical_scaled = scaler.fit_transform(X_numerical)

X_combined = hstack([X_ngrams, X_numerical_scaled])

y = data['label_simplified'].values

print(f"Feature matrix shape: {X_combined.shape}")
print(f"  - N-gram features: {X_ngrams.shape[1]}")
print(f"  - Numerical features: {X_numerical.shape[1]}")
print(f"  - Total features: {X_combined.shape[1]}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_combined, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nTraining Hybrid Multinomial Naive Bayes model...")
model = MultinomialNB(
    alpha=0.1  
)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)
test_acc = accuracy_score(y_test, y_pred_test)

print("\n=== Accuracy Summary (Hybrid: N-grams + Numerical Features) ===")
print(f"Training Accuracy   : {train_acc:.4f}")
print(f"Validation Accuracy : {val_acc:.4f}")
print(f"Test Accuracy       : {test_acc:.4f}")

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test, labels=['ENG', 'FIL', 'OTH'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['ENG', 'FIL', 'OTH'],
            yticklabels=['ENG', 'FIL', 'OTH'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Hybrid Multinomial Naive Bayes (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("\nConfusion matrix saved as 'confusion_matrix.png'")

with open("trained_model_hybrid.pkl", "wb") as f:
    pickle.dump((model, vectorizer, scaler), f)

print("Hybrid Multinomial Naive Bayes model, vectorizer, and scaler saved to 'trained_model.pkl'")

print("\n" + "="*60)
print("CURRENT MODEL PERFORMANCE")
print("="*60)
print(f"Hybrid Multinomial Naive Bayes: {test_acc:.2%}")
print("="*60)