# Memorization guard

Code: *planned* (index built from `data/interim/*.clean.txt`; checked before TTS)

## Why

Language models repeat text they've seen. Even though the generator is trained on *synthetic*
robot-voice sets, the prompts and skeletons are derived from real comedians' bits, and the base
LLM may already know famous lines. A comedy agent repeating another comic's joke is plagiarism
by comedy's own standards, and copyright exposure besides.

## How

1. Build a set of hashed **word 8-grams** (lowercased, punctuation stripped) from every
   transcript's clean text. Only hashes are stored, never the text.
2. Before a generated line is spoken, hash its 8-grams. **Any overlap → reject** and regenerate.
3. Log rejections (hash count, not text) to see if one premise keeps producing copies.

## Why 8-grams

Short n-grams (3–5 words) match everyday phrases ("I don't know about you"), so the guard
would reject everything. 8 consecutive identical words almost never happen by chance in
conversational English but reliably catch a reused punchline. Tune with the false-positive
rate on synthetic sets: if it rejects too much, use 10.

## Limits

It catches verbatim reuse, not paraphrase. A paraphrased famous bit gets through, so premise
skeletons are kept abstract, and spot-checks look for recognizable bits.
