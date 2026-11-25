"""JUDGE: rank one prompt's candidate jokes from pairwise "which is funnier?" calls
(Bradley-Terry + adaptive LUCB pair selection). See judge/README.md.

    bt.py       Bradley-Terry math: fit, standard errors, P(i beats j), Borda
    judges.py   SimulatedJudge (tests), LLMJudge (llama3.1:8b on Ollama), LogitJudge (A/B logits)
    select.py   which pairs to ask: all pairs, warm start, adaptive LUCB
    rank.py     rank(): ask the judge, fit, stop; rank_from_probs() for bulk scores; write_judgments()
    records.py  Comparison (one judge call) and Ranking (one prompt's result)
    run_llm.py  command line: rank a joke list with the LLM judge
"""
from .bt import fit_bt, win_prob  # NEW
from .judges import LLMJudge, LogitJudge, SimulatedJudge  # NEW
from .rank import rank, rank_from_probs, write_judgments
from .records import Comparison, Ranking  # NEW

__all__ = ["rank", "rank_from_probs", "Ranking", "Comparison", "SimulatedJudge", "LLMJudge", "LogitJudge",
           "fit_bt", "win_prob", "write_judgments"]
