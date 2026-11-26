# Tonight's context: headlines, the market and one line

**Where it fits:** the writer's input, before the safety filter. The writer is told a little about today, so a bit can be about something that is happening.
**Used in:** `backend/app/tonight.py` and the `TONIGHT` setting, the pre-show panel on the operator screen, Exp 33 in the [live decision notebook](../../notebooks/decisions-live.ipynb), decision D-117.

## Why

A comic's best material is what everyone already has in their head tonight: the weather, the news, the mood of the market. A fixed topic list (D-116, a separate branch) cannot know that. A robot that says "so the stock market is down again" sounds like it is in the room.

## What the writer gets

| Piece | Where it comes from | How it is used |
|---|---|---|
| Market line | The S&P 500's last two daily closes (Stooq), written as one sentence | Background in every bit |
| Headline gist | Google News top stories, rewritten as one nameless sentence | Bits 2 and 6 take the next unused one as their topic |
| One line about tonight | The operator, optional, for example "it's snowing" | Background in every bit |

## What stays out

- **Names.** One Claude Haiku call rewrites each headline with generic words ("a GPU company", "a tech CEO", "a senator"). A cheap rule then rejects any gist that still has a capitalized name in the middle of a sentence. The no-named-people rule is unchanged.
- **Tragedies.** The same call labels each headline, and death, disaster, crime and war are dropped before anyone sees them. Politics and health stay, because they are what a room talks about, but they still go through the normal safety check.
- **Anything the operator did not approve.** The pre-show panel lists the gists, all unticked. Only ticked gists, the market line, and the one typed line (also safety-checked) reach the writer.

A headline is used up only when its bit is actually played, the same rule as D-116's material. If the fetch fails, or nothing is approved, the show runs exactly as before.

## What we found

Stooq now answers scripts with a "prove you are a browser" page instead of the CSV, so the market line may be missing. Google News works. See D-117.

## How we test it

Exp 33 is run by hand. You approve each gist the filter kept and flag any with a name: at most 20% rejected and no names. Then 10 pairs of bits, with and without tonight's context, shown blind: the new one must be funnier in at least 5 of 10. One day of headlines and 10 pairs are small samples, so a pass means "not clearly worse". See [D-117](../decisions.md).
