#!/usr/bin/env python3
"""
Produces:
 - takeshi_parsed_words.csv  (word-level table)
 - takeshi_preprocessing_log.txt (summary + examples)
"""

import re
import csv
from collections import defaultdict
from pathlib import Path

INPUT = Path("data/takeshi.txt")
OUT_CSV = Path("data/takeshi_parsed_words.csv")
OUT_LOG = Path("logs/takeshi_preprocessing_log.txt")

# Regex to match lines like "<f1r.P1.1;H> rest of line"
TAG_LINE_RE = re.compile(r'^\s*<(?P<tag>[^>]+)>\s*(?P<content>.*)$')
CLEAN_TOKEN_RE_STRIP = re.compile(r'^[^A-Za-z0-9\.]+|[^A-Za-z0-9\.]+$')
COLLAPSE_DOTS_RE = re.compile(r'\.{2,}')

def clean_token(original: str):
    """
    Return (cleaned_token_or_None, unified_token_or_None, has_markup_bool, ambiguity_notes_list)
    """
    notes = []
    has_markup = False

    if original is None:
        return (None, None, False, ["none_input"])

    # Mark presence of simple markup
    if "*" in original:
        has_markup = True
        notes.append("contains_*")

    if re.search(r'[^A-Za-z0-9\.\*]', original):
        # non-standard chars beyond letters, digits, dot, asterisk
        notes.append("non_standard_chars")
        has_markup = True

    cleaned = original

    # Remove asterisks used as uncertainty markup
    if "*" in cleaned:
        cleaned = cleaned.replace("*", "")

    # Collapse repeated dots (e.g., "...." -> ".")
    cleaned = COLLAPSE_DOTS_RE.sub(".", cleaned)

    # Strip leading/trailing non-alphanum except keep '.' inside tokens
    cleaned = CLEAN_TOKEN_RE_STRIP.sub("", cleaned)

    # Normalize empty -> None
    if cleaned == "":
        cleaned = None

    unified = cleaned.lower() if cleaned is not None else None

    # Ambiguity heuristics
    amb = []
    if has_markup:
        amb.append("markup_removed")
    if cleaned is None:
        amb.append("empty_after_clean")
    elif cleaned != original and (cleaned.lower() != original.lower()):
        amb.append("changed_by_cleaning")
    if original.count(".") >= 4:
        amb.append("many_dots")

    return (cleaned, unified, has_markup, amb)


def parse_file(path: Path):
    rows = []
    stats = defaultdict(int)
    folio_set = set()
    ambiguous_examples = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line.strip():
                stats["blank_lines"] += 1
                continue

            m = TAG_LINE_RE.match(line)
            if not m:
                # Line without a tag: treat as UNGROUPED
                tag = None
                content = line.strip()
                folio = "UNGROUPED"
                scribal = None
                meta_left = None
                stats["untagged_lines"] += 1
            else:
                tag = m.group("tag").strip()
                content = m.group("content").strip()
                if ";" in tag:
                    left, scribal = tag.split(";", 1)
                    scribal = scribal.strip()
                else:
                    left = tag
                    scribal = None
                meta_left = left.strip()
                if "." in meta_left:
                    folio = meta_left.split(".", 1)[0]
                else:
                    folio = meta_left
                folio_set.add(folio)

            stats["total_tagged_lines"] += 1 if tag else 0

            # If content empty, write a line entry with no words
            if content == "":
                rows.append({
                    "folio": folio,
                    "tag": tag,
                    "meta_left": meta_left,
                    "scribal": scribal,
                    "line_text": content,
                    "word_index": None,
                    "original_word": None,
                    "cleaned_word": None,
                    "unified_word": None,
                    "has_markup": False,
                    "ambiguity_notes": None,
                })
                continue

            # small tidy: remove stray leading "H>" if present (from some transcripts)
            if content.startswith("H>"):
                content = content[2:].lstrip()
                stats["removed_leading_H>"] += 1

            words = content.split()
            for wi, w in enumerate(words, start=1):
                original = w
                cleaned, unified, has_markup, amb = clean_token(original)

                if has_markup:
                    stats["words_with_markup"] += 1
                stats["total_words"] += 1

                if amb and len(ambiguous_examples) < 40:
                    ambiguous_examples.append((original, cleaned, amb))

                rows.append({
                    "folio": folio,
                    "tag": tag,
                    "meta_left": meta_left,
                    "scribal": scribal,
                    "line_text": content,
                    "word_index": wi,
                    "original_word": original,
                    "cleaned_word": cleaned,
                    "unified_word": unified,
                    "has_markup": has_markup,
                    "ambiguity_notes": ";".join(amb) if amb else None,
                })

    return rows, stats, folio_set, ambiguous_examples


def write_csv(rows, out_path: Path):
    if not rows:
        # write an empty CSV with header
        header = ["folio","tag","meta_left","scribal","line_text","word_index",
                  "original_word","cleaned_word","unified_word","has_markup","ambiguity_notes"]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
        return

    header = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_log(stats, folio_set, ambiguous_examples, out_path: Path, rows):
    lines = []
    lines.append(f"Source file: {INPUT}")
    lines.append(f"Total word-level rows: {len(rows)}")
    lines.append(f"Total tokens processed: {stats.get('total_words', 0)}")
    lines.append(f"Blank lines skipped: {stats.get('blank_lines', 0)}")
    lines.append(f"Untagged lines: {stats.get('untagged_lines', 0)}")
    lines.append(f"Lines with removed leading 'H>': {stats.get('removed_leading_H>', 0)}")
    lines.append(f"Words containing markup (e.g., '*'): {stats.get('words_with_markup', 0)}")
    lines.append(f"Unique folios found: {len(folio_set)}")
    lines.append("")

    # Top unified tokens (simple frequency)
    lines.append("Top 20 unified tokens (by frequency):")
    freq = defaultdict(int)
    for r in rows:
        u = r.get("unified_word")
        if u:
            freq[u] += 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    for token, cnt in top:
        lines.append(f"  {token:30} {cnt}")
    lines.append("")

    lines.append("Examples of tokens flagged as ambiguous (original -> cleaned ; notes):")
    for orig, cleaned, notes in ambiguous_examples[:40]:
        lines.append(f"  {orig:30} -> {str(cleaned):30} ; {notes}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        return

    rows, stats, folio_set, ambiguous_examples = parse_file(INPUT)
    write_csv(rows, OUT_CSV)
    write_log(stats, folio_set, ambiguous_examples, OUT_LOG, rows)
    print("Done.")
    print(f"Word-level CSV: {OUT_CSV}")
    print(f"Preprocessing log: {OUT_LOG}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
