import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np

# 1. Load the data
df = pd.read_csv('data/takeshi_parsed_words.csv')

# 2. Preprocessing & Tokenization
# The 'cleaned_word' column contains lines of text with words separated by dots (e.g., "word1.word2.word3")
df_clean = df.dropna(subset=['cleaned_word'])

all_tokens = []
for line in df_clean['cleaned_word']:
    # Split by '.' and remove empty strings resulting from consecutive/trailing dots
    words = [w for w in str(line).split('.') if w]
    all_tokens.extend(words)

# 3. Compute Basic Statistics
total_tokens = len(all_tokens)
unique_types = len(set(all_tokens))
word_lengths = [len(w) for w in all_tokens]
total_characters = sum(word_lengths)
avg_word_length = np.mean(word_lengths)

# Character Frequencies
all_text = "".join(all_tokens)
char_counts = collections.Counter(all_text)

# 4. Language Comparisons
# One-letter words
one_letter_words = [w for w in all_tokens if len(w) == 1]
count_one_letter = len(one_letter_words)

# Vowel Proportion
# Common Voynich "vowels" in EVA transcription are generally considered to be: o, a, y, e (and sometimes i)
voynich_vowels = {'o', 'a', 'y', 'e', 'i'}
vowel_count = sum(count for char, count in char_counts.items() if char in voynich_vowels)
vowel_proportion = vowel_count / total_characters if total_characters > 0 else 0

# 5. Output Results
print(f"--- Statistics ---")
print(f"Total Word Tokens: {total_tokens}")
print(f"Unique Word Types: {unique_types}")
print(f"Total Characters: {total_characters}")
print(f"Average Word Length: {avg_word_length:.2f}")
print(f"One-letter Words: {count_one_letter} ({count_one_letter/total_tokens:.2%} of total)")
print(f"Vowel Proportion (a, e, i, o, y): {vowel_proportion:.2%}")
print(f"Top 5 Characters: {char_counts.most_common(5)}")

# 6. Plotting
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram of Word Lengths
axes[0].hist(word_lengths, bins=range(1, 15), align='left', rwidth=0.8, color='#5DADE2', edgecolor='black')
axes[0].set_title('Distribution of Word Lengths')
axes[0].set_xlabel('Length (characters)')
axes[0].set_ylabel('Frequency')

# Bar chart of Character Frequencies (Top 20)
sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:20]
chars, counts = zip(*sorted_chars)

axes[1].bar(chars, counts, color='#58D68D', edgecolor='black')
axes[1].set_title('Top 20 Character Frequencies')
axes[1].set_xlabel('Character')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('logs/voynich_stats.png')