# Voynich Manuscript Decoder Challenge

## Goal

Use AI to decode the Voynich Manuscript, one of the most mysterious books in the world, written in an unknown script and language.

## Tasks

- Build a pipeline that ingests transcribed Voynich text (EVA or Takahashi transcription)
- Use LLMs, embeddings, or custom models to find patterns, possible meanings, or linguistic structures
- Try to match parts of the text with known languages, glyph frequencies, or hypothesized semantics

## Requirements

- Use AI reasoning to explore unknown language or construct hypotheses
- Provide clear logs of your process
- Explain why you believe your approach may uncover meaning

## Bonus Features

- Visual overlay of decoded terms on manuscript images
- Model fine-tuned on similar ciphered texts
- Timeline of symbol usage evolution across manuscript pages

## Anotações

### Transliterações

#### EVA - Extensible Voynich Alphabet

Seu design faz parte de um esquema maior que inclui:

- A necessidade de representar o texto completo do manuscrito Voynich, incluindo partes esquecidas e alguns dos textos fora do padrãos de parágrafos
- O alfabeto deveria ser um super conjunto do FSG (First Study Group), Currier e Frogguy, de forma que seja possível fazer a conversão de qualquer transliteração para o EVA e vice-versa
- Uma forma clara de identificar caracteres raros/estranhos
- Padronização do formato de transliteração incluindo uma forma padrão de identificação da localização de todos os itens do texto no manuscrito o qual foi dado o nome de locus
- A capacidade de estruturar uma True Type Font que representa o texto do manuscrito Voynich com precisão

Para isso, os caracteres foram divididos em dois conjuntos: Basic EVA, que se trata do conjunto de caracteres minúsculos identificados, e o Extended EVA, que representa os caracteres raros/estranhos.

Os caracteres raros/estranhos podem ser classificados como:

- Caracter único raro ou incomum;
- Ligações de caracteres do conjunto Basic EVA
- Ligações incluindo outros caracteres raros/estranho

#### Takaheshi

Transliteração que utilizou o EVA para cobrir, aproximadamente, 97% de todas as linhas do Voynich Manuscript.

Dessa forma, devido ao fato de cobrir quase todos o manuscrito, a transliteração escolhida para o projeto é a Takaheshi.

## Processos e Etapas

Para começar o projeto, eu inicialmente faço o download do arquivo TXT [takeshi](./data/takeshi.txt) que se refere a transliteração original do manuscrito de Voynich feita pelo Takeshi Takaheshi, segundo o [site](https://voynich.nu/transcr.html) que reune todas as informações existente sobre esse manuscrito.

Após isso, eu realizo a limpeza dos dados e transformação deles para uma tabela CSV usando um [script](./parse_takeshi.py) python que contém as seguintes colunas:

- folio -> código da página do manuscrito
- tag -> a tag inteira entre <> que contém os metados das palavras do mauscrito
- meta_left -> tag antes do `;`
- scribal -> o que vêm depois de `;` na tag original
- line_text -> conteúdo orginal da linha da transliteração
- word_index -> index da palavra dentro da linha
- original_word -> palavra como ela está na transliteração
- cleaned_word -> palavras após tratamento (remoção de marcações simples como `*`, cortando caracteres não alfabéticos no final e começo da palavra e colapsando pontos repetidos)
- unified_word -> forma limpa e minúscula das palavras
- has_markup -> valor booleano que representa se havia qualquer tipo de marcação na palavra
- ambiguity_notes

### Análise Exploratória

#### Estatísticas básicas


