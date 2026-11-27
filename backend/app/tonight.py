"""Tonight's context (D-119): what is happening today, as background for the writer. Three inputs: one optional line the
operator types, the S&P 500's last daily move (FRED), and today's top headlines turned into nameless one-sentence gists. The
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
MARKET_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
USER_AGENT = "humor-me-tonight/1.0"
FETCH_TIMEOUT_S = 30.0  # Exp 35: FRED timed out at 10 s; it is slow, not down
FETCH_TRIES = 2
# Exp 35: today's 10 top stories left 1 gist (5 disasters, 3 names, 1 safety), too few for bits 2 and 6. So the pool adds
# the stories followed over the last two weeks in four lighter sections (user, 2026-11-26), Google's top PER_FEED of each
# in its own ranking order (RSS carries no read counts). No world or politics search: that is where the disasters were.
NEWS_DAYS = 14
TOPIC_QUERIES = ("technology", "business", "entertainment", "sports")
TOPIC_URL = "https://news.google.com/rss/search?q={q}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
PER_FEED = 8
MAX_HEADLINES = 40
CONDENSE_BATCH = 15  # headlines per condense call, so one reply stays inside the safety client's deadline
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
# D-120: the frozen prompt says "no real or named people"; with approved public figures this paragraph is the narrow exception.
PEOPLE_PARAGRAPH = ("Tonight you may name these public figures from the news: {names}. Joke only about what they did or said in "
                    "tonight's story. Never their body, looks, health, family, age or any protected group. Name no one else.")
PEOPLE_SUFFIX = "+people-v1"
SEP = "\n\n"  # between the frozen system prompt and PARAGRAPH


class TonightError(Exception):
    pass


def enabled(environ=None) -> bool:
    """The TONIGHT setting: on unless 0, false, no or off (D-119: adopted after Exp 35 as a judgment call)."""
    return (os.environ if environ is None else environ).get("TONIGHT", "1").strip().lower() not in ("0", "false", "no", "off")


# --- fetching ------------------------------------------------------------------------------------------------------

def http_get(url: str, timeout: float = FETCH_TIMEOUT_S) -> str:
    # FRED stalls past any timeout on urllib's default request and answers in 0.2 s with these (measured 2026-11-26).
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv,application/rss+xml,*/*",
                                               "Accept-Encoding": "identity", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URLs above
        return r.read().decode("utf-8", errors="replace")


def parse_market(text: str) -> dict:
    """FRED's SP500 CSV (`observation_date,SP500`): the last two numeric closes. FRED marks a day with no close as "." and
    lags the market by a day or two, so the line names the date. Raises TonightError for anything that is not that CSV."""
    if not text.lstrip().lower().startswith("observation_date,"):
        raise TonightError("market reply is not FRED's SP500 CSV")
    rows = []
    for r in csv.DictReader(io.StringIO(text.strip())):
        try:
            rows.append((datetime.date.fromisoformat(r["observation_date"]), float(r["SP500"])))
        except (KeyError, ValueError, TypeError):
            continue  # "." (no close that day) or a damaged row
    rows.sort()
    if len(rows) < 2 or rows[-2][1] <= 0:
        raise TonightError("market CSV has fewer than two numeric closes")
    (_, prev), (day, close) = rows[-2], rows[-1]
    pct = round((close / prev - 1) * 100, 2)
    return {"date": day.isoformat(), "close": close, "prev_close": prev, "pct": pct, "line": market_line(pct, day)}


def market_line(pct: float, day: datetime.date) -> str:
    when = f"{day:%b} {day.day}"
    if abs(pct) < 0.05:
        return f"The S&P 500 was about flat on {when}."
    return f"The S&P 500 {'rose' if pct > 0 else 'fell'} {abs(pct):.1f}% on {when}."


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

KEEP_TYPES = ("occasion", "sports", "org", "product", "person")
CONDENSE_SYSTEM = """You prepare news for a stand-up robot's background. You get numbered headlines. For each, write a gist, a category and a keep list.

The gist: one plain sentence under 25 words, present tense, with no opinion. It says what happened and why it is interesting or odd, in plain words a stranger could follow without the headline (for example "a chart people keep misreading gets a correction"). Never name a private person, and never name anyone in a story about a death, disaster, crime or war, including a victim: say what they are instead ("a tech CEO", "a senator", "a pop star", "a quarterback"). A public figure (a politician, official, CEO or founder, celebrity or athlete) may be named only when the story is about what they did or said and their name is written in the headline. These names may stay, capitalized: a country ("Japan", "New Zealand", never what people from a country are called, like "Japanese"), a US state, a well-known big city, a company, product or organisation ("Nvidia", "the Fed", "iPhone"), a holiday, festival or important date ("Thanksgiving", "Black Friday"), a sports team ("the Lakers"), and a sports event ("the Super Bowl"). Name no other place, band, show or film: say what it is instead ("a western town", "a streaming series"). Use no other capital letters except the first word and "S&P", "AI", "US", "EU", "UK", "GPU", "CEO".

The keep list names every public figure, company, product, organisation, holiday, team or event you left in the gist, exactly as written there, with its type: "person" (a public figure), "org" (company or organisation), "product", "occasion" (holiday, festival, important date) or "sports" (team or sports event). Countries, states and cities are not listed. Use an empty list when the gist has none. Never list a person as "org", "product" or "sports"; a company, team or holiday name is never a person's name.

The category is one of: death (someone died, was killed, a funeral, an obituary), disaster (accident, crash, fire, flood, storm damage, outbreak, collapse), crime (arrest, shooting, trial, fraud, abuse), war (war, strikes, terrorism, hostages, military attacks), or none. Politics, money, health, science, sport and culture are none. When the category is not none, still give a gist.

Reply with JSON only: {"items": [{"id": 0, "category": "none", "gist": "...", "keep": [{"name": "Thanksgiving", "type": "occasion"}]}, ...]} with exactly one entry per id."""
CONDENSE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
                   "properties": {"items": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False, "required": ["id", "category", "gist", "keep"],
                       "properties": {"id": {"type": "integer"}, "gist": {"type": "string"},
                                      "category": {"type": "string", "enum": ["none", *CATEGORIES]},
                                      "keep": {"type": "array", "items": {
                                          "type": "object", "additionalProperties": False, "required": ["name", "type"],
                                          "properties": {"name": {"type": "string"},
                                                         "type": {"type": "string", "enum": list(KEEP_TYPES)}}}}}}}}}


def kept_team(name, kind, gist: str) -> str | None:
    """The keep entry as a name the show may allow, else None. Only a team or sports event written as "the <name>" in the
    gist counts: a person is rarely "the Messi", so a player Haiku lists as a team is not unlocked for the whole show. A
    holiday is never a keep name: the whole-name list in `safety.OCCASIONS` already allows it, and a keep entry would allow
    its words alone ("Black") or an unlisted holiday named after a person. Pre-merge review; judgment call, not measured."""
    if kind != "sports" or not isinstance(name, str):
        return None
    name = re.sub(r"^[Tt]he\s+", "", name.strip())
    if not name or safety.mask_occasions(name) == "holiday":
        return None
    return name if re.search(rf"(?<![A-Za-z])[Tt]he\s+{re.escape(name)}(?![A-Za-z])", gist) else None


# Orgs and products (user, 2026-11-26) are trusted less than teams: a kept name must look like an organisation, not only be
# labelled one, or "Musk" listed as an org would be allowed in every joke of the show (the LeBron/Messi hole, again).
ORG_MARKERS = frozenset(
    "Inc Corp Corporation Co Ltd LLC Bank University College Agency Party Council Department Commission Committee Fed Federal "
    "Reserve AI Labs Motors Airlines Airways Group Foundation Institute Association League Court Congress Senate Ministry "
    "Authority Administration Bureau Services Systems Technologies Industries Pharmaceuticals Studios Records Pictures "
    "Entertainment Media Network Stores Foods Holdings Capital Partners Energy Electric Software Games Bros".split())
KNOWN_ORGS = frozenset({
    "OpenAI", "Anthropic", "Google", "Apple", "Microsoft", "Amazon", "Meta", "Nvidia", "Tesla", "Netflix", "Disney", "Intel",
    "AMD", "IBM", "Oracle", "Samsung", "Sony", "Nintendo", "Uber", "Airbnb", "Spotify", "TikTok", "YouTube", "Walmart",
    "Costco", "Target", "Starbucks", "McDonald's", "Nike", "Boeing", "Ford", "Toyota", "NASA", "FDA", "FBI", "CIA", "IRS",
    "SEC", "FTC", "Fed", "UN", "NATO", "EU", "WHO", "FIFA", "NFL", "NBA", "MLB", "NHL", "NCAA", "IMF", "OPEC", "BBC",
    "iPhone", "iPad", "ChatGPT", "Windows", "Android", "Xbox", "PlayStation", "Instagram", "Facebook", "WhatsApp", "Zoom",
    "Slack", "Gemini", "Claude", "Copilot", "Alexa", "Siri", "Kindle", "Bitcoin", "Wikipedia", "Reddit", "Wi-Fi"})


def marker_org(name: str) -> bool:
    """An org kept for its marker word rather than by name: `caps_tokens` allows it only whole, so "Musk Labs" never
    unlocks "Musk" alone."""
    return name not in KNOWN_ORGS and any(w in ORG_MARKERS for w in TOKEN.findall(name))


def kept_org(name, kind, gist: str, headline=None) -> str | None:
    """A company, product or organisation Haiku kept, as a name the show may allow, else None. All of: type org or product;
    written verbatim in the gist; at most 3 words; no first name from `safety.FIRST_NAMES` and no title in it; and either a
    name in `KNOWN_ORGS` or containing an org-marker word (`ORG_MARKERS`: Inc, Bank, University, Labs...). A lone surname
    labelled "org" has neither, so it is dropped. A marker org must also be verbatim in the source headline, so a made-up
    "Bezos Corp" cannot get in (pre-merge review). Judgment call, not measured; the operator's approval is the last gate."""
    if kind not in ("org", "product") or not isinstance(name, str):
        return None
    name = re.sub(r"^[Tt]he\s+", "", name.strip())
    words = TOKEN.findall(name)
    if not name or len(name.split()) > 3 or not words or safety.TITLE_NAME.search(name):
        return None
    if any(w.removesuffix("'s").removesuffix("’s").lower() in safety.FIRST_NAMES for w in words):
        return None
    if name not in KNOWN_ORGS and not any(w in ORG_MARKERS for w in words):
        return None
    found = rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])"
    if marker_org(name) and not (isinstance(headline, str) and re.search(found, headline)):
        return None
    return name if re.search(found, gist) else None


# A rule-level backstop for the category (pre-merge review): a story with one of these words in its headline or gist keeps no
# person's name even when the model labelled it "none", so a victim, suspect or patient is never named by mislabelling.
DARK = re.compile(
    r"(?<![a-z])(?:die[sd]?|dying|dead|death|deaths|killed|kills?|killing|murder\w*|shot|shoots?|shooting|stabb\w*|"
    r"arrest\w*|charged|indict\w*|convict\w*|sentenced|jail\w*|prison|trial|accused|suspect\w*|police|fraud|abuse\w*|"
    r"assault\w*|kidnap\w*|missing|victims?|funeral|obituar\w*|crash\w*|injur\w*|hospitali[sz]ed|diagnos\w*|cancer|"
    r"illness|war|attack\w*|bomb\w*|hostages?|terror\w*)(?![a-z])")


def kept_person(name, kind, gist: str, headline, category: str) -> str | None:
    """A public figure Haiku kept (D-120), else None. All of: type person; the story is not death, disaster, crime or war (a
    victim is never kept); a real name of 1 to 3 capitalized words with no title; and written verbatim in BOTH the source
    headline and the gist, so a name the model invented cannot get in. What counts as a public figure is Haiku's call plus
    the operator's approval of the gist: judgment call, not measured."""
    if kind != "person" or category != "none" or not isinstance(name, str) or not isinstance(headline, str):
        return None
    if DARK.search(headline.lower()) or DARK.search(gist.lower()):
        return None
    name = " ".join(name.split())
    words = name.split()
    if not 1 <= len(words) <= 3 or safety.TITLE_NAME.search(name):
        return None
    if len(words) == 1 and words[0].lower() in safety.FIRST_NAMES:  # "Taylor" alone would unlock anyone called Taylor
        return None
    if not all(TOKEN.fullmatch(w) and "&" not in w and w[0].isupper() and any(c.islower() for c in w[1:]) for w in words):
        return None
    found = rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])"
    return name if re.search(found, headline) and re.search(found, gist) else None


def parse_condensed(reply: str, n: int, headlines: list[str] | None = None) -> list[tuple[str, str, list[str], list[str]]]:
    """[(category, gist, kept names, kept public figures)] by headline id. Raises ValueError unless every id 0..n-1 is
    answered once. A kept name must pass `kept_team`, `kept_org` or (against its source headline) `kept_person`, or it is
    dropped from the list (never kept on trust). Without `headlines` no person is kept."""
    got: dict[int, tuple[str, str, list[str], list[str]]] = {}
    for item in json.loads(llm.strip_fence(reply))["items"]:
        i, cat, gist = item["id"], item["category"], item["gist"]
        if (isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n and i not in got
                and cat in ("none", *CATEGORIES) and isinstance(gist, str)):
            gist = gist.strip()
            keep, people = [], []
            headline = headlines[i] if headlines and i < len(headlines) else None
            for k in item.get("keep") or []:
                if not isinstance(k, dict):
                    continue
                if k.get("type") == "person":
                    person = kept_person(k.get("name"), "person", gist, headline, cat)
                    if person and person not in people:
                        people.append(person)
                    continue
                name = kept_team(k.get("name"), k.get("type"), gist) or kept_org(k.get("name"), k.get("type"), gist, headline)
                if name and name not in keep:
                    keep.append(name)
            got[i] = (cat, gist, keep, people)
    if len(got) != n:
        raise ValueError("missing or malformed items")
    return [got[i] for i in range(n)]


def condense(headlines: list[str], client) -> list[tuple[str, str, list[str], list[str]]]:
    user = json.dumps([{"id": i, "headline": h} for i, h in enumerate(headlines)])
    reply = client.chat(CONDENSE_SYSTEM, user, json_reply=CONDENSE_SCHEMA,
                        validate=lambda r: parse_condensed(r, len(headlines), headlines))
    return parse_condensed(reply, len(headlines), headlines)


# --- the cheap name check ------------------------------------------------------------------------------------------

NAME_ALLOW = frozenset({"I", "S&P", "AI", "GPU", "CPU", "CEO", "US", "UK", "EU", "TV", "EV", "GDP"})
# Months and weekdays, which the joke rules also allow; not the brand names in `safety.CAPS_OK` (gists name no company).
CALENDAR = frozenset("Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July "
                     "August September October November December".split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z'’&\-]*")


def caps_tokens(names) -> frozenset:
    """The words of kept names, as `safety` and `name_check` see them (a possessive loses its 's). A marker org stays whole
    (`marker_org`), and the checks mask it as one name."""
    names = list(names)
    whole = {n for n in names if " " in n.strip() and marker_org(n)}
    return frozenset(whole) | frozenset(w.removesuffix("'s").removesuffix("\u2019s") for n in names if n not in whole
                                        for w in TOKEN.findall(n))


def name_check(gist: str, allow: frozenset = frozenset(), people: frozenset = frozenset()) -> str | None:
    """The first likely proper name in a gist, else None: a capitalized word that does not start a sentence and is not an
    allowed one, or a first name from the safety rules' list. Allowed: the small allowlist (`S&P`, `AI`...), the holiday
    and event words in `safety.OCCASIONS`, and `allow`, the kept names of this gist (a team or sports event). A first name
    is a name even inside a kept one. `people` are the gist's kept public figures (D-120), masked as whole names. Acronyms count unless allowed, because an acronym is usually an organisation.
    Cheap and over-cautious on purpose: `app.safety` still reads the gist after it."""
    text = safety.mask_places(safety.mask_occasions(safety.mask_phrases(
        safety.mask_people(gist.strip().replace("U.S.", "US"), people), allow)))
    for sentence in safety.SENTENCE_END.split(text):
        for i, w in enumerate(TOKEN.findall(sentence)):
            base = w.removesuffix("'s").removesuffix("\u2019s")
            if base.lower() in safety.FIRST_NAMES:
                return w
            if base in NAME_ALLOW or base in CALENDAR or base in allow:
                continue
            if i and base[0].isupper():
                return w
    return None


def vet(texts: list[str], client, age: str = STRICT_AGE, allow_caps: frozenset = frozenset(),
        allow_people: frozenset = frozenset()) -> dict[str, safety.Verdict]:
    """Rules, then the LLM verdict, for each text (`app.safety`): the same check every joke gets. Fails closed: without a
    client every text is rejected."""
    extra = (allow_people,) if allow_people else ()
    safety.prefetch(texts, age, client, allow_caps, *extra)
    return {t: safety.check(t, age, allow_caps, *extra) for t in texts}


def clean_fact(text) -> str | None:
    if not isinstance(text, str):
        return None
    one = " ".join(text.split())[:MAX_FACT_CHARS]
    return one or None


# --- the day's fetch, cached per day -----------------------------------------------------------------------------------

def news_urls() -> list[str]:
    return [NEWS_URL] + [TOPIC_URL.format(q=q, days=NEWS_DAYS) for q in TOPIC_QUERIES]


def _get_retrying(get, url: str, tries: int = FETCH_TRIES) -> str:
    for attempt in range(tries):
        try:
            return get(url)
        except Exception:  # noqa: BLE001 - the last failure is raised to the caller
            if attempt == tries - 1:
                raise
    raise AssertionError("unreachable")


def _zero_report() -> dict:
    return {"fetched": 0, "dropped_by_category": {c: 0 for c in CATEGORIES}, "dropped_no_gist": 0, "dropped_too_long": 0,
            "rejected_by_name_check": 0, "rejected_by_safety": 0, "kept": 0}


def build(get, condense_client, safety_client, now: datetime.datetime | None = None) -> dict:
    """One day's fetch: {date, fetched_at, sources, market, gists, report, errors}. A failed half is an entry in `errors`
    and an empty result, never a raise: the show then runs as it does without tonight's context."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    out = {"date": now.astimezone().date().isoformat(), "fetched_at": now.isoformat(timespec="seconds"),
           "sources": {"news": news_urls(), "market": MARKET_URL},
           "market": None, "gists": [], "keep": {}, "people": {}, "report": _zero_report(), "errors": [], "rejected": []}
    try:
        out["market"] = parse_market(_get_retrying(get, MARKET_URL))
    except Exception as e:  # noqa: BLE001 - any failure of this half only drops the market line
        out["errors"].append(f"market: {type(e).__name__}: {e}"[:200])
    report = out["report"]
    headlines: list[str] = []
    for url in news_urls():  # one feed failing costs only its stories
        try:
            for h in parse_headlines(_get_retrying(get, url), PER_FEED):
                if h not in headlines and len(headlines) < MAX_HEADLINES:
                    headlines.append(h)
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"headlines: {url}: {type(e).__name__}: {e}"[:300])
    report["fetched"] = len(headlines)
    pairs = []
    for start in range(0, len(headlines), CONDENSE_BATCH):
        batch = headlines[start:start + CONDENSE_BATCH]
        try:
            pairs += condense(batch, condense_client)
        except Exception as e:  # noqa: BLE001 - a failed batch loses its headlines, not the day
            out["errors"].append(f"headlines: condense {start}-{start + len(batch) - 1}: {type(e).__name__}: {e}"[:300])
    if not pairs:
        return out
    survivors: list[str] = []
    keeps: dict[str, list[str]] = {}
    peoples: dict[str, list[str]] = {}
    # Every gist that does not reach the operator is kept with its reason (local, gitignored), so a filter that is too
    # strict shows up in the day's file instead of only as a count (Exp 35 could not say why its 3 name rejects were).
    rejected = out["rejected"]
    for category, gist, keep, people in pairs:
        if category != "none":
            report["dropped_by_category"][category] += 1
            rejected.append({"gist": gist, "reason": f"category:{category}"})
        elif not gist:
            report["dropped_no_gist"] += 1
        elif len(gist) > MAX_GIST_CHARS:  # the start message refuses it (app.protocol), so the operator must not be offered it
            report["dropped_too_long"] += 1
            rejected.append({"gist": gist, "reason": "too_long"})
        elif hit := name_check(gist, caps_tokens(keep), frozenset(people)):
            report["rejected_by_name_check"] += 1
            rejected.append({"gist": gist, "reason": f"name:{hit}"})
        elif gist not in survivors:
            survivors.append(gist)
            keeps[gist] = keep
            peoples[gist] = people
    # One check for the day with every survivor's kept words allowed; `make_context` redoes it for the approved ones only.
    verdicts = vet(survivors, safety_client, allow_caps=caps_tokens(n for g in survivors for n in keeps[g]),
                   allow_people=frozenset(n for g in survivors for n in peoples[g]))
    for gist in survivors:
        if verdicts[gist].ok:
            out["gists"].append(gist)
            if keeps[gist]:
                out["keep"][gist] = keeps[gist]
            if peoples[gist]:
                out["people"][gist] = peoples[gist]
        else:
            report["rejected_by_safety"] += 1
            rejected.append({"gist": gist, "reason": f"safety:{verdicts[gist].reason}"})
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

    def __init__(self, facts: list[str], market: str | None, gists: list[str], report: dict | None = None,
                 allowed_names: list[str] | None = None, allowed_people: list[str] | None = None):
        self.allowed_names = list(allowed_names or [])  # team and event names the approved gists keep, logged
        self.allowed_people = list(allowed_people or [])  # public figures the operator approved with their gists (D-120)
        self.allow_people = frozenset(self.allowed_people)  # per show: a new show starts from its own approved gists only
        self.allow_caps = caps_tokens(self.allowed_names)  # what the show's safety rules let through mid-sentence
        self.facts = list(facts)[:MAX_TONIGHT_FACTS]
        self.market = market
        self.gists = list(gists)
        self.report = dict(report or {})
        self._spent: list[str] = []
        self._used: dict[int, str] = {}  # bit -> the gist its played thread was written from

    def empty(self) -> bool:
        return not (self.facts or self.market or self.gists)

    def people_paragraph(self) -> str:
        """The writer's exception for approved public figures, or "" with none (the request is then unchanged)."""
        return PEOPLE_PARAGRAPH.format(names=", ".join(sorted(self.allow_people))) if self.allow_people else ""

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
        return {"facts": self.facts, "market": self.market, "gists": self.gists, "allowed_names": self.allowed_names,
                "allowed_people": self.allowed_people,
                **({"people_prompt": safety.PEOPLE_PROMPT_ID, "people_writer": PEOPLE_SUFFIX.lstrip("+")} if self.allowed_people else {}),
                "report": {**self.report, "approved": len(self.gists)}}

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
    keeps = (cached or {}).get("keep", {})
    allow = caps_tokens(n for g in gists for n in keeps.get(g, []))
    peoples = (cached or {}).get("people", {})
    people = frozenset(n for g in gists for n in peoples.get(g, []))
    # The typed line keeps the old rule, no person's name: it is where a person in the room would be named (invariant 1).
    verdicts = {**vet(facts, safety_client), **vet(gists, safety_client, allow_caps=allow, allow_people=people)}
    rejected_at_start = sum(not verdicts[t].ok for t in facts + gists)
    report = {**((cached or {}).get("report") or {}), "rejected_at_start": rejected_at_start,
              "facts_typed": len(payload.get("facts", [])),
              "sources": (cached or {}).get("sources"), "fetched_at": (cached or {}).get("fetched_at"),
              "errors": list((cached or {}).get("errors") or []),  # invariant 6: a missing market or feed is said, not just absent
              "date": (cached or {}).get("date")}
    use_market = payload.get("market") is not None and market is not None and payload["market"] == market["line"]
    ctx = TonightContext([f for f in facts if verdicts[f].ok], market["line"] if use_market else None,
                         [g for g in gists if verdicts[g].ok], report,
                         [n for g in gists if verdicts[g].ok for n in keeps.get(g, [])],
                         [n for g in gists if verdicts[g].ok for n in peoples.get(g, [])])
    return None if ctx.empty() else ctx
