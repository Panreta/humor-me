# Tonight's context: headlines, the market and one line

**Where it fits:** the writer's input, before the safety filter. The writer is told a little about today, so a bit can be about something that is happening.
**Used in:** `backend/app/tonight.py` and the `TONIGHT` setting, the pre-show panel on the operator screen, Exp 35 in the [live decision notebook](../../notebooks/decisions-live.ipynb), decision D-119.

## Why

A comic's best material is what everyone already has in their head tonight: the weather, the news, the mood of the market. A fixed topic list (D-117) cannot know that. A robot that says "so the stock market is down again" sounds like it is in the room.

## What the writer gets

| Piece | Where it comes from | How it is used |
|---|---|---|
| Market line | The S&P 500's last two daily closes (FRED), written as one sentence that names the date | Background in every bit |
| Headline gist | Google News: today's top stories and two weeks of technology, business, entertainment and sports, each rewritten as one nameless sentence | Bits 2 and 6 take the next unused one as their topic |
| One line about tonight | The operator, optional, for example "it's snowing" | Background in every bit |

## What stays out

- **Names.** One Claude Haiku call rewrites each headline with generic words ("a GPU company", "a tech CEO", "a senator"). A cheap rule then rejects any gist that still has a capitalized name in the middle of a sentence. The no-named-people rule is unchanged. The one exception: holidays, festivals, important dates and sports teams or events may stay capitalized ("Thanksgiving", "the Super Bowl", a team). The big holidays on a fixed list are always fine. A team name is allowed only when the summary writes it as "the Lakers" and the summary is approved; then the jokes may repeat it. A player is never kept as a team.
- **Tragedies.** The same call labels each headline, and death, disaster, crime and war are dropped before anyone sees them. Politics and health stay, because they are what a room talks about, but they still go through the normal safety check.
- **Anything the operator did not approve.** The pre-show panel lists the gists, all unticked; you can tick at most 10. Only ticked gists, the market line, and the one typed line (also safety-checked) reach the writer.

A headline is used up only when its bit is actually played, the same rule as D-117's material. If the fetch fails, or nothing is approved, the show runs exactly as before.

## What we found

FRED's SP500 file lags the market by a day or two, so the market line names its date. We first tried Stooq, which now answers scripts with a "prove you are a browser" page. Google News works. See D-119.

## How we test it

Exp 35 is run by hand. You approve each gist the filter kept and flag any with a name: at most 20% rejected and no names. Then 10 pairs of bits, with and without tonight's context, shown blind: the new one must be funnier in at least 5 of 10. One day of headlines and 10 pairs are small samples, so a pass means "not clearly worse". See [D-119](../decisions.md).

**Result (Nov 26):** the new bits felt like tonight in 9 of 10 pairs and tied on funniness. Only 1 of 10 headlines survived the filter that day, too few to judge the filter, so the news now also comes from two weeks of lighter sections. It is on by default as the user's judgment call, not because it passed; `TONIGHT=0` turns it off. Numbers: [metrics.md](../metrics.md#tonights-context-for-each-bit-exp-35-2026-11-26).
