import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import Counter

# 1. Load Data
df = pd.read_csv('takeshi_parsed_words.csv')
df_clean = df.dropna(subset=['cleaned_word'])

# 2. Preprocess & Tokenize
sentences = []
all_tokens = []
for line in df_clean['cleaned_word']:
    words = [w for w in str(line).split('.') if w]
    if words:
        sentences.append(words)
        all_tokens.extend(words)

# 3. Build Embeddings (SVD on PPMI Matrix)
# Parameters
vocab_size = 600  # Focus on top frequent words for clearer clusters
window_size = 2   # Context window (left/right)

# Build Vocabulary
counts = Counter(all_tokens)
vocab = [w for w, c in counts.most_common(vocab_size)]
word_to_id = {w: i for i, w in enumerate(vocab)}

# Build Co-occurrence Matrix
co_matrix = np.zeros((vocab_size, vocab_size))
for sent in sentences:
    sent_ids = [word_to_id[w] for w in sent if w in word_to_id]
    for i, center_id in enumerate(sent_ids):
        start = max(0, i - window_size)
        end = min(len(sent_ids), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                context_id = sent_ids[j]
                co_matrix[center_id, context_id] += 1

# PPMI Calculation (Positive Pointwise Mutual Information)
total_count = np.sum(co_matrix)
row_sums = np.sum(co_matrix, axis=1)
col_sums = np.sum(co_matrix, axis=0)
expected = np.outer(row_sums, col_sums) / total_count

with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log(co_matrix / expected)
ppmi_matrix = np.maximum(pmi, 0) # Clip negatives
ppmi_matrix = np.nan_to_num(ppmi_matrix)

# Create Dense Vectors via SVD (Simulates Word2Vec)
svd = TruncatedSVD(n_components=50, random_state=42)
word_vectors = svd.fit_transform(ppmi_matrix)

# 4. t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
vectors_2d = tsne.fit_transform(word_vectors)

# 5. Plotting
plt.figure(figsize=(15, 12))

# Color mapping by first letter (Morphological grouping)
first_letters = [w[0] for w in vocab]
unique_starts = sorted(list(set(first_letters)))
color_map = {l: i for i, l in enumerate(unique_starts)}
colors = [color_map[l] for l in first_letters]

plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colors, cmap='tab20', alpha=0.7, edgecolors='k', linewidth=0.3)

# Annotate words
for i, w in enumerate(vocab):
    # Annotate top 40 frequent words + specific clusters of interest
    if i < 40 or any(sub in w for sub in ['daiin', 'shedy', 'chol', 'ol']):
        plt.text(vectors_2d[i, 0]+0.2, vectors_2d[i, 1]+0.2, w, fontsize=9, alpha=0.9)

# Legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab20(color_map[l]/len(unique_starts)), label=l) 
           for l in unique_starts if l in ['o', 'y', 'd', 's', 'q', 'c', 't', 'p']]
plt.legend(handles=handles, title="Starts With", loc='upper right')

plt.title('Voynich Word Embeddings: t-SNE Projection of Semantic Clusters')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('voynich_embeddings_tsne.png')