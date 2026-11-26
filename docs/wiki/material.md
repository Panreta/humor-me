# Material: topics and character facts

**Where it fits:** the writer's input, before the safety filter. Each bit gets one topic and two facts about the robot, so the jokes have something to be about.
**Used in:** `backend/app/material.py` and the `MATERIAL` setting, Exp 33 in the [live decision notebook](../../notebooks/decisions-live.ipynb), decision D-117.

## The problem

The writer was told almost nothing: one sentence about the robot, the mode, the room, and the premises already told this show. That list starts empty every show. So the small model kept reaching for what it knows best, which is the robot itself: battery, processor, memory, shutting down. In the 35 logged shows, 28% of played jokes named robot hardware.

## Why comics need specifics

A comedian does not walk on stage with "be funny". They bring a point of view, and a pile of specific things to notice. "My smart fridge talks about me behind my back" is a bit. "Technology is confusing" is not. Two things follow:

- **Specifics make a character.** A robot with a mute button its operator uses on calls, a wish for a houseplant, and opinions about fonts sounds like someone. A robot with "a battery" sounds like a device.
- **Topic variety keeps a set fresh.** A listener who has heard the airport-security bit does not want it again next week. Comedians retire a bit once the room has heard it.

## What we do

| Piece | What it is | Rule |
|---|---|---|
| Character sheet | About 30 specific facts about the robot's life ([character-v1](../../backend/app/material/character-v1.yaml)) | 2 per bit, the least used this show first |
| Topic list | About 110 everyday situations the robot takes an outsider's look at ([topics-v1](../../backend/app/material/topics-v1.yaml)) | 1 per bit, never one a show on this laptop has already used |
| Robot-life topics | The topics marked `from: character`: the robot's own life as the whole bit | About 1 bit in 3 (bits 1, 4 and 7) |

A topic or fact counts as used only when its bit is actually played. A bit written on a wrong guess of the next mode is thrown away without using anything up (see D-115). Teen rooms skip the few topics marked adult. The writer is also told not to build the bit on its own hardware. Every played bit logs its topic and facts in a `material` row.

## What is not done yet

Typed facts about tonight, a few news headlines and the day's stock move, all checked for safety and approved by the operator before the show, come in a later PR (D-117, option B).

## How we test it

Exp 33 writes 5 shows with and without material and compares them. The bars: hardware share of at most 15% and clearly below the old prompt's, and the new bit picked as funnier in at least 10 of 20 blind pairs. 20 pairs only detect large differences. The setting stays off until both pass. Result: hardware bits fell from 88% to 12% (bar met), but the new bit won only 6 of 20 funniness pairs (bar 10), so it stays off. See [D-117](../decisions.md).
