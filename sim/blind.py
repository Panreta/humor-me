"""Text as a blind rater sees it. Only qwen3:4b-instruct writes curly quotes, em dashes and ellipses (D-105, Exp 24), so
shown as written they would tell the user and the judge which bits are its. Display only: stored bits stay as written."""

PLAIN = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": "-", "…": "..."})


def plain(text: str) -> str:
    return " ".join(text.translate(PLAIN).split())  # " - " next to an existing space would leave a double one
