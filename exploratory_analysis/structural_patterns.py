import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('data/takeshi_parsed_words.csv')
# FIX 1: Use .copy() to ensure we have a standalone DataFrame, not a view
df_clean = df.dropna(subset=['cleaned_word', 'folio']).copy()

# 2. Define Mappings
def get_folio_number(folio_str):
    match = re.search(r'(\d+)', str(folio_str))
    return int(match.group(1)) if match else 0

def get_section(folio_num):
    if 1 <= folio_num <= 66: return 'Herbal'
    elif 67 <= folio_num <= 73: return 'Astronomical'
    elif 75 <= folio_num <= 84: return 'Biological'
    elif 85 <= folio_num <= 102: return 'Pharmaceutical'
    elif 103 <= folio_num <= 116: return 'Recipes'
    return 'Unknown'

def get_currier(folio_num, section):
    # Simplified Currier A/B mapping
    if section == 'Herbal':
        return 'A' if 1 <= folio_num <= 25 else 'B'
    elif section == 'Biological':
        return 'B'
    return 'Mixed' # Astro, Pharma, Recipes are often mixed or 'Language C'

# Apply Mappings
df_clean['folio_num'] = df_clean['folio'].apply(get_folio_number)
df_clean['section'] = df_clean['folio_num'].apply(get_section)
df_clean['currier'] = df_clean.apply(lambda x: get_currier(x['folio_num'], x['section']), axis=1)

# Explode to individual words
all_rows = []
for idx, row in df_clean.iterrows():
    words = [w for w in str(row['cleaned_word']).split('.') if w]
    for w in words:
        all_rows.append({'folio_num': row['folio_num'], 'section': row['section'], 
                         'currier': row['currier'], 'word': w})
df_words = pd.DataFrame(all_rows)

# 3. Dialect Analysis: Compare A vs B
df_ab = df_words[df_words['currier'].isin(['A', 'B'])]
counts = df_ab.groupby(['currier', 'word']).size().reset_index(name='count')
totals = df_ab.groupby('currier')['word'].count().to_dict()
counts['freq'] = counts.apply(lambda x: x['count'] / totals[x['currier']], axis=1)

# Pivot to find differences
df_pivot = counts.pivot(index='word', columns='currier', values='freq').fillna(0)
df_pivot['diff'] = df_pivot['A'] - df_pivot['B']

# Top Discriminators
top_a = df_pivot.sort_values('diff', ascending=False).head(10) # Prefer A
top_b = df_pivot.sort_values('diff', ascending=True).head(10)  # Prefer B

# Marker Words for Plotting
word_a = top_a.index[0] 
word_b = top_b.index[0] 

# 4. Visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2)

# Plot 1: Top Words by Section
section_counts = df_words.groupby(['section', 'word']).size().reset_index(name='count')
# FIX 2: Use sort_values + head instead of apply to avoid FutureWarning
top_sec = section_counts.sort_values(['section', 'count'], ascending=[True, False]).groupby('section').head(5)

sns.barplot(data=top_sec, x='count', y='word', hue='section', dodge=False, ax=fig.add_subplot(gs[0, 0]))
plt.title('Top Words per Section')

# Plot 2: Marker Word Shift (Folio Timeline)
folio_counts = df_words.groupby(['folio_num', 'word']).size().unstack(fill_value=0)
page_totals = df_words.groupby('folio_num').size()
valid_folios = page_totals[page_totals > 0].index
# Normalize frequencies
freq_a = folio_counts.loc[valid_folios].get(word_a, 0) / page_totals.loc[valid_folios]
freq_b = folio_counts.loc[valid_folios].get(word_b, 0) / page_totals.loc[valid_folios]

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(valid_folios, freq_a, label=f'Type A ("{word_a}")', color='blue', alpha=0.7)
ax2.plot(valid_folios, freq_b, label=f'Type B ("{word_b}")', color='red', alpha=0.7)
ax2.set_title(f'Dialect Shift: "{word_a}" vs "{word_b}" across Folios')
ax2.set_xlabel('Folio Number')
ax2.legend()
# Add section lines
for x in [1, 67, 75, 85, 103]: ax2.axvline(x, color='gray', linestyle='--', alpha=0.3)

# Plot 3: Discriminator Comparison
div_data = pd.concat([top_a, top_b]).reset_index().melt(id_vars='word', value_vars=['A', 'B'], var_name='Dialect', value_name='Frequency')
sns.barplot(data=div_data, x='word', y='Frequency', hue='Dialect', ax=fig.add_subplot(gs[1, :]), palette={'A':'blue', 'B':'red'})
plt.title('Strongest Dialect Markers (A vs B)')

plt.tight_layout()
plt.savefig('logs/voynich_structural_analysis.png')