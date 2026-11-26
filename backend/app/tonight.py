"""Tonight's context (D-117): what is happening today, as background for the writer. Three inputs: one optional line the
operator types, the S&P 500's last daily move, and today's top headlines turned into nameless one-sentence gists. The
operator approves them on the operator screen before the show; only approved items reach the writer.

Names stay generic ("a GPU company", "a senator"): the no-real-or-named-people rule is unchanged. Death, disaster, crime
and war are dropped before the operator sees anything. Every gist and typed line then goes through the safety check
(`app.safety`, rules then LLM, fail closed) at the strictest age, because nobody knows the room yet.

Used means PLAYED, like `app.material`: a headline is spent only when its bit's thread was told (`TonightContext.played`),
so a bit written on a guess that is thrown away (D-115) spends nothing, and drawing twice for one bit gives one answer.
Network calls send nothing about the user: a plain generic User-Agent and no cookies."""
import csv
import datetime
import io
import json
import os
import pathlib
import re
import urllib.request
import xml.etree.ElementTree as ET

from app import llm, safety

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "tonight"
NEWS_URL = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
MARKET_URL = "https://stooq.com/q/d/l/?s=^spx&i=d"
USER_AGENT = "humor-me-tonight/1.0"
FETCH_TIMEOUT_S = 10.0
MAX_HEADLINES = 10
MAX_TONIGHT_FACTS = 1  # the user's call: headlines and the market already add content, so the typed line is optional
MAX_FACT_CHARS = 120
MAX_GISTS = 10
MAX_GIST_CHARS = 200
HEADLINE_EVERY = 4  # bits where bit % 4 == 2 (bits 2 and 6) take a headline as the topic; they miss material's bits 1, 4, 7
HEADLINE_REMAINDER = 2
STRICT_AGE = "teens"  # the room is unknown at approval time, so the strictest rules apply
CATEGORIES = ("death", "disaster", "crime", "war")
PARAGRAPH = ("The user message gives tonight's context as background: a market line and possibly one fact about tonight. "
             "Use them only where they make a line funnier, never as the whole bit. When it also gives a topic that is a "
             "headline, build the bit on that topic, seen the way this robot sees it; keep it vague, name nobody, and "
             "do not make the bit about the robot's own hardware.")
SUFFIX = "+tonight-v1"
SEP = "\n\n"  # between the frozen system prompt and PARAGRAPH


class TonightError(Exception):
    pass


def enabled(environ=None) -> bool:
    """The TONIGHT setting: off unless 1, true, yes or on."""
    return (os.environ if environ is None else environ).get("TONIGHT", "0").strip().lower() in ("1", "true", "yes", "on")


# --- fetching ------------------------------------------------------------------------------------------------------

def http_get(url: str, timeout: float = FETCH_TIMEOUT_S) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URLs above
        return r.read().decode("utf-8", errors="replace")


def parse_market(text: str) -> dict:
    """Stooq's daily CSV (Date,Open,High,Low,Close,Volume): the last two closes. Raises TonightError for anything else,
    including the browser check Stooq serves to scripts (an HTML page), so the show then runs without a market line."""
    if not text.lstrip().lower().startswith("date,"):
        raise TonightError("market reply is not the daily CSV (the site may be asking for a browser check)")
    rows = []
    for r in csv.DictReader(io.StringIO(text.strip())):
        try:
            rows.append((r["Date"], float(r["Close"])))
        except (KeyError, ValueError, TypeError):
            continue
    rows.sort()
    if len(rows) < 2 or rows[-2][1] <= 0:
        raise TonightError("market CSV has fewer than two usable closes")
    (_, prev), (date, close) = rows[-2], rows[-1]
    pct = round((close / prev - 1) * 100, 2)
    return {"date": date, "close": close, "prev_close": prev, "pct": pct, "line": market_line(pct)}


def market_line(pct: float) -> str:
    if abs(pct) < 0.05:
        return "The S&P 500 closed about flat at its last close."
    return f"The S&P 500 closed {'up' if pct > 0 else 'down'} {abs(pct):.1f}% at its last close."


def parse_headlines(xml_text: str, limit: int = MAX_HEADLINES) -> list[str]:
    """Item titles of a Google News RSS feed with the trailing ' - Source' removed, deduplicated, in feed order."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise TonightError("headline feed is not valid XML") from e
    out: list[str] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        title = title.rsplit(" - ", 1)[0].strip() if " - " in title else title
        if title and title not in out:
            out.append(title)
        if len(out) >= limit:
            break
    if not out:
        raise TonightError("headline feed had no items")
    return out


# --- the Haiku step: nameless gists, off-limits categories dropped -----------------------------------------------------

CONDENSE_SYSTEM = """You prepare news for a stand-up robot's background. You get numbered headlines. For each, write a gist and a category.

The gist: one plain sentence under 20 words, present tense, with no opinion. Never name a person, company, product, band, team, show or place: say what they are instead ("a GPU company", "a tech CEO", "a senator", "a western state", "a streaming service"). Use no capital letters except the first word and "S&P", "AI", "US", "EU", "UK", "GPU", "CEO".

The category is one of: death (someone died, was killed, a funeral, an obituary), disaster (accident, crash, fire, flood, storm damage, outbreak, collapse), crime (arrest, shooting, trial, fraud, abuse), war (war, strikes, terrorism, hostages, military attacks), or none. Politics, money, health, science, sport and culture are none. When the category is not none, still give a gist.

Reply with JSON only: {"items": [{"id": 0, "category": "none", "gist": "..."}, ...]} with exactly one entry per id."""
CONDENSE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
                   "properties": {"items": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False, "required": ["id", "category", "gist"],
                       "properties": {"id": {"type": "integer"}, "gist": {"type": "string"},
                                      "category": {"type": "string", "enum": ["none", *CATEGORIES]}}}}}}


def parse_condensed(reply: str, n: int) -> list[tuple[str, str]]:
    """[(category, gist)] by headline id. Raises ValueError unless every id 0..n-1 is answered once."""
    got: dict[int, tuple[str, str]] = {}
    for item in json.loads(llm.strip_fence(reply))["items"]:
        i, cat, gist = item["id"], item["category"], item["gist"]
        if (isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n and i not in got
                and cat in ("none", *CATEGORIES) and isinstance(gist, str)):
            got[i] = (cat, gist.strip())
    if len(got) != n:
        raise ValueError("missing or malformed items")
    return [got[i] for i in range(n)]


def condense(headlines: list[str], client) -> list[tuple[str, str]]:
    user = json.dumps([{"id": i, "headline": h} for i, h in enumerate(headlines)])
    reply = client.chat(CONDENSE_SYSTEM, user, json_reply=CONDENSE_SCHEMA,
                        validate=lambda r: parse_condensed(r, len(headlines)))
    return parse_condensed(reply, len(headlines))


# --- the cheap name check ------------------------------------------------------------------------------------------

NAME_ALLOW = frozenset({"I", "S&P", "AI", "GPU", "CPU", "CEO", "US", "UK", "EU", "TV", "EV", "GDP"})
TOKEN = re.compile(r"[A-Za-z][A-Za-z'’&\-]*")


def name_check(gist: str) -> str | None:
    """The first likely proper name in a gist, else None: a capitalized word that does not start a sentence and is not
    allowlisted, or a first name from the safety rules' list. Acronyms count unless allowlisted, because an acronym is
    usually an organisation. Cheap and over-cautious on purpose: `app.safety` still reads the gist after it."""
    text = gist.strip().replace("U.S.", "US")
    for sentence in safety.SENTENCE_END.split(text):
        for i, w in enumerate(TOKEN.findall(sentence)):
            base = w.removesuffix("'s").removesuffix("’s")
            if base in NAME_ALLOW:
                continue
            if base.lower() in safety.FIRST_NAMES or (i and base[0].isupper()):
                return w
    return None


def vet(texts: list[str], client, age: str = STRICT_AGE) -> dict[str, safety.Verdict]:
    """Rules, then the LLM verdict, for each text (`app.safety`): the same check every joke gets. Fails closed: without a
    client every text is rejected."""
    safety.prefetch(texts, age, client)
    return {t: safety.check(t, age) for t in texts}


def clean_fact(text) -> str | None:
    if not isinstance(text, str):
        return None
    one = " ".join(text.split())[:MAX_FACT_CHARS]
    return one or None


# --- the day's fetch, cached per day -----------------------------------------------------------------------------------

def _zero_report() -> dict:
    return {"fetched": 0, "dropped_by_category": {c: 0 for c in CATEGORIES}, "dropped_no_gist": 0,
            "rejected_by_name_check": 0, "rejected_by_safety": 0, "kept": 0}


def build(get, condense_client, safety_client, now: datetime.datetime | None = None) -> dict:
    """One day's fetch: {date, fetched_at, sources, market, gists, report, errors}. A failed half is an entry in `errors`
    and an empty result, never a raise: the show then runs as it does without tonight's context."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    out = {"date": now.astimezone().date().isoformat(), "fetched_at": now.isoformat(timespec="seconds"), "sources": {"news": NEWS_URL, "market": MARKET_URL},
           "market": None, "gists": [], "report": _zero_report(), "errors": []}
    try:
        out["market"] = parse_market(get(MARKET_URL))
    except Exception as e:  # noqa: BLE001 - any failure of this half only drops the market line
        out["errors"].append(f"market: {type(e).__name__}: {e}"[:200])
    report = out["report"]
    try:
        headlines = parse_headlines(get(NEWS_URL))
        report["fetched"] = len(headlines)
        pairs = condense(headlines, condense_client)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"headlines: {type(e).__name__}: {e}"[:200])
        return out
    survivors: list[str] = []
    for category, gist in pairs:
        if category != "none":
            report["dropped_by_category"][category] += 1
        elif not gist:
            report["dropped_no_gist"] += 1
        elif name_check(gist):
            report["rejected_by_name_check"] += 1
        elif gist not in survivors:
            survivors.append(gist)
    verdicts = vet(survivors, safety_client)
    for gist in survivors:
        if verdicts[gist].ok:
            out["gists"].append(gist)
        else:
            report["rejected_by_safety"] += 1
    report["kept"] = len(out["gists"])
    return out


def cache_path(date: str, directory: pathlib.Path | None = None) -> pathlib.Path:
    return (directory or DATA) / f"{date}.json"


def load_or_build(build_fn, *, refresh: bool = False, directory: pathlib.Path | None = None,
                  today: str | None = None) -> dict:
    """The cached fetch for today, else `build_fn()`. A fetch where both halves failed is not cached."""
    today = today or datetime.date.today().isoformat()
    path = cache_path(today, directory)
    if not refresh and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # a damaged cache is a fetch
    result = build_fn()
    if result["market"] is not None or result["gists"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


# --- the show's side ---------------------------------------------------------------------------------------------------

class TonightContext:
    """One per show: the approved facts, market line and gists, and which gists were spent."""

    def __init__(self, facts: list[str], market: str | None, gists: list[str], report: dict | None = None):
        self.facts = list(facts)[:MAX_TONIGHT_FACTS]
        self.market = market
        self.gists = list(gists)
        self.report = dict(report or {})
        self._spent: list[str] = []
        self._used: dict[int, str] = {}  # bit -> the gist its played thread was written from

    def empty(self) -> bool:
        return not (self.facts or self.market or self.gists)

    def background(self) -> dict | None:
        return {"facts": self.facts, "market": self.market} if (self.facts or self.market) else None

    def draw(self, bit: int) -> str | None:
        """The next unused approved gist for a headline bit, else None. A pure function of what has been played."""
        if bit % HEADLINE_EVERY != HEADLINE_REMAINDER:
            return None
        return next((g for g in self.gists if g not in self._spent), None)

    def played(self, bit: int) -> None:
        """The bit's thread was told: its gist is spent. A second call for the same bit does nothing."""
        if bit in self._used:
            return
        if (gist := self.draw(bit)) is not None:
            self._spent.append(gist)
            self._used[bit] = gist

    def summary(self) -> dict:
        """The `tonight` log row: what was approved and where it came from (invariant 6)."""
        return {"facts": self.facts, "market": self.market, "gists": self.gists, "report": {**self.report, "approved": len(self.gists)}}

    def use_row(self, bit: int) -> dict | None:
        return {"gist": self._used[bit]} if bit in self._used else None


def make_context(payload: dict, cached: dict | None, safety_client) -> TonightContext | None:
    """The show's context from the operator's approved payload ({facts, market, gists}), checked again here: the typed
    line goes through the safety check for the first time, and every gist and the market line must be what today's fetch
    produced (the browser is not trusted to invent them). `cached` is today's `build` result, which supplies the counts,
    sources and fetch time for the log. None when nothing survives."""
    known = set((cached or {}).get("gists", []))
    market = (cached or {}).get("market")
    facts = [f for f in (clean_fact(x) for x in payload.get("facts", [])) if f][:MAX_TONIGHT_FACTS]
    gists = [g for g in payload.get("gists", []) if g in known]
    verdicts = vet(facts + gists, safety_client)
    rejected_at_start = sum(not verdicts[t].ok for t in facts + gists)
    report = {**((cached or {}).get("report") or {}), "rejected_at_start": rejected_at_start,
              "facts_typed": len(payload.get("facts", [])),
              "sources": (cached or {}).get("sources"), "fetched_at": (cached or {}).get("fetched_at"),
              "date": (cached or {}).get("date")}
    use_market = payload.get("market") is not None and market is not None and payload["market"] == market["line"]
    ctx = TonightContext([f for f in facts if verdicts[f].ok], market["line"] if use_market else None,
                         [g for g in gists if verdicts[g].ok], report)
    return None if ctx.empty() else ctx
