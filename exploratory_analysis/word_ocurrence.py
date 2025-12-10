import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# 1. Load Data
df = pd.read_csv('data/takeshi_parsed_words.csv')
df_clean = df.dropna(subset=['cleaned_word'])

# 2. Word Position Analysis
# We count how often a word appears at the start, middle, or end of a line.
word_position_counts = defaultdict(lambda: {'start': 0, 'middle': 0, 'end': 0, 'total': 0})
all_words_flat = []

for line in df_clean['cleaned_word']:
    # Words are separated by dots.
    words = [w for w in str(line).split('.') if w]
    if not words:
        continue
    
    all_words_flat.extend(words)
    
    # Analyze position in line
    if len(words) == 1:
        # Single word line: counts as start and end
        w = words[0]
        word_position_counts[w]['start'] += 1
        word_position_counts[w]['end'] += 1
        word_position_counts[w]['total'] += 1
    else:
        # First word
        word_position_counts[words[0]]['start'] += 1
        word_position_counts[words[0]]['total'] += 1
        # Last word
        word_position_counts[words[-1]]['end'] += 1
        word_position_counts[words[-1]]['total'] += 1
        # Middle words
        for w in words[1:-1]:
            word_position_counts[w]['middle'] += 1
            word_position_counts[w]['total'] += 1

# Convert Word Stats to DataFrame
pos_data = []
for word, counts in word_position_counts.items():
    if counts['total'] > 10: # Filter for frequent words
        pos_data.append({
            'word': word,
            'total': counts['total'],
            'start_freq': counts['start'] / counts['total'],
            'end_freq': counts['end'] / counts['total']
        })

df_pos = pd.DataFrame(pos_data)
top_starts = df_pos.sort_values('start_freq', ascending=False).head(10)
top_ends = df_pos.sort_values('end_freq', ascending=False).head(10)

# 3. Glyph Position Analysis (Start/End of WORD) & N-grams
glyph_pos_counts = defaultdict(lambda: {'start': 0, 'middle': 0, 'end': 0, 'total': 0})
all_text = "".join(all_words_flat) # Joined text for transition matrix

for w in all_words_flat:
    chars = list(w)
    if not chars: continue
    
    # Start char
    glyph_pos_counts[chars[0]]['start'] += 1
    glyph_pos_counts[chars[0]]['total'] += 1
    
    # End char
    if len(chars) > 1:
        glyph_pos_counts[chars[-1]]['end'] += 1
        glyph_pos_counts[chars[-1]]['total'] += 1
    
    # Middle chars
    if len(chars) > 2:
        for c in chars[1:-1]:
            glyph_pos_counts[c]['middle'] += 1
            glyph_pos_counts[c]['total'] += 1
    elif len(chars) == 1:
        # Single char word: counts as start and end
        glyph_pos_counts[chars[0]]['end'] += 1

glyph_data = []
for char, counts in glyph_pos_counts.items():
    if counts['total'] > 50: # Filter rare chars
        glyph_data.append({
            'char': char,
            'total': counts['total'],
            'start_prop': counts['start'] / counts['total'],
            'end_prop': counts['end'] / counts['total'],
            'middle_prop': counts['middle'] / counts['total']
        })
df_glyph_pos = pd.DataFrame(glyph_data).sort_values('char')

# 4. Character Transition Matrix (Bigram Probability)
# This unsupervised method groups letters by their transition properties (e.g. vowels vs consonants)
common_chars = sorted([c for c, count in Counter(all_text).items() if count > len(all_text)*0.001])
char_idx = {c: i for i, c in enumerate(common_chars)}
n_chars = len(common_chars)
trans_matrix = np.zeros((n_chars, n_chars))

for i in range(len(all_text) - 1):
    c1, c2 = all_text[i], all_text[i+1]
    if c1 in char_idx and c2 in char_idx:
        trans_matrix[char_idx[c1], char_idx[c2]] += 1

# Normalize to probabilities
with np.errstate(divide='ignore', invalid='ignore'):
    trans_prob = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)
    trans_prob = np.nan_to_num(trans_prob)

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 16))

# Plot 1: Top Line-Start Words
sns.barplot(data=top_starts, x='start_freq', y='word', ax=axes[0,0], palette='Blues_d')
axes[0,0].set_title('Words Preferring Line-Start Position')

# Plot 2: Top Line-End Words
sns.barplot(data=top_ends, x='end_freq', y='word', ax=axes[0,1], palette='Reds_d')
axes[0,1].set_title('Words Preferring Line-End Position')

# Plot 3: Glyph Position Heatmap
heatmap_data = df_glyph_pos.set_index('char')[['start_prop', 'middle_prop', 'end_prop']]
sns.heatmap(heatmap_data, cmap='viridis', ax=axes[1,0])
axes[1,0].set_title('Glyph Position Preferences')

# Plot 4: Transition Matrix
sns.heatmap(trans_prob, xticklabels=common_chars, yticklabels=common_chars, cmap='inferno', ax=axes[1,1])
axes[1,1].set_title('Character Transition Probabilities (Bigram Model)')

plt.tight_layout()
plt.savefig('logs/voynich_linguistic_analysis.png')