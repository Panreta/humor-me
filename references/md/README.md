# Reference Papers (Markdown)

Extracted from the PDFs in `references/` via pymupdf4llm. Figures are dropped; tables are kept where detected. Each file's first line names its source PDF.

| File | Paper | Source PDF | Key topics |
|---|---|---|---|
| [humorgen-cognitive-synergy-persona-distillation.md](humorgen-cognitive-synergy-persona-distillation.md) | HumorGen: Cognitive Synergy for Humor Generation in LLMs via Persona-Based Distillation (ACL submission) | `1241_HumorGen_Cognitive_Synerg.pdf` | Mixture-of-Thought, 6 comedic personas, distillation to 7B, DPO vs Offline GRPO |
| [automating-humor-template-extraction-infilling.md](automating-humor-template-extraction-infilling.md) | Automating Humor: Joke Generation Using Template Extraction and Infilling (Goel et al., ICON 2024) | `2024.icon-1.51.pdf` | Joke templates, masked infilling |
| [structured-thought-leaps-humor-iclr2025.md](structured-thought-leaps-humor-iclr2025.md) | Innovative Thinking, Infinite Humor: Humor Research of LLMs through Structured Thought Leaps (Wang et al., ICLR 2025, arXiv 2410.10370) | `2410.10370v2.pdf` | Structured thought leaps for humor (builds on CLoT), judge/reward models, DPO |
| [deep-rl-from-human-preferences.md](deep-rl-from-human-preferences.md) | Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017) | `Deep reinforcement learning from human preferences.pdf` | Reward modeling from pairwise preferences, RLHF foundations |

Known extraction artifacts: section headings often appear as bold text rather than `#` headings; HumorGen had ACL line numbers stripped, so some hyphenated compounds are joined (e.g. `highquality`).

## Regenerating

```sh
pip install -r requirements.txt
python scripts/pdf_to_md.py references/new-paper.pdf references/md/descriptive-name.md
```
