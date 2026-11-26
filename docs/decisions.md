# Decisions

What we decided and why, one line each. Click an ID for the full record (options, numbers, evidence).

| ID | Date | Decision |
| --- | --- | --- |
| [D-01](agent/decisions-detail.md#d-01) | Oct 20 | Audio only, no camera: faster, and the robot can't joke about how people look. |
| [D-02](agent/decisions-detail.md#d-02) | Oct 20 | Learn from laughs per joke; report the whole-show rating. Ratings are too rare to learn from, and raw laugh counts can be gamed. |
| [D-03](agent/decisions-detail.md#d-03) | Oct 20 | One robot personality; a bandit picks which comedy style to use. Switching whole personas felt incoherent. |
| [D-04](agent/decisions-detail.md#d-04) | Oct 20 | Start with an AI judge; replace it later with a scorer trained on our own audience votes. |
| [D-05](agent/decisions-detail.md#d-05) | Oct 20 | Main data is our own show logs; rJokes and published methods get us started. |
| [D-06](agent/decisions-detail.md#d-06) | Oct 20 | Research questions: does adapting beat a fixed style? Does a trained scorer beat the AI judge? |
| [D-07](agent/decisions-detail.md#d-07) | Oct 20 | First milestone is the Dec 15 course demo, with the RL part proven in a simulator. Date later moved (D-44). |
| [D-08](agent/decisions-detail.md#d-08) | Oct 20 | Before a show the operator sets venue, energy and age group. Audio can't tell age. |
| [D-09](agent/decisions-detail.md#d-09) | Oct 20 | Teen audiences always get a stricter safety filter. |
| [D-10](agent/decisions-detail.md#d-10) | Oct 20 | The system writes the next bit based on how the room is reacting. |
| [D-11](agent/decisions-detail.md#d-11) | Oct 20 | Comedy transcripts teach structure and laugh timing only; we never copy their jokes. |
| [D-12](agent/decisions-detail.md#d-12) | Oct 20 | Transcript laughs are scored 0 to 4; "laughed or not" is the main label. |
| [D-13](agent/decisions-detail.md#d-13) | Oct 20 | Opener and closer are positions in the set, not kinds of bits. |
| [D-14](agent/decisions-detail.md#d-14) | Oct 20 | Only transcripts that mark laughs often are used for laugh labels. |
| [D-15](agent/decisions-detail.md#d-15) | Oct 21 | Applause without laughter doesn't count as a laugh. |
| [D-16](agent/decisions-detail.md#d-16) | Oct 21 | Transcripts split into two datasets: with laugh labels, and structure only. |
| [D-17](agent/decisions-detail.md#d-17) | Oct 21 | Callbacks are something the planner can choose. Later deferred (D-46). |
| [D-18](agent/decisions-detail.md#d-18) | Oct 21 | The laugh predictor learns from single lines, because passage length alone gave the answer away. |
| [D-19](agent/decisions-detail.md#d-19) | Oct 21 | Use free Gemini to label transcripts, sending as little text as possible. Replaced by D-20. |
| [D-20](agent/decisions-detail.md#d-20) | Oct 21 | Send each whole set to Gemini in one call; it answers with line numbers, never quotes. |
| [D-21](agent/decisions-detail.md#d-21) | Oct 21 | A bit is the smallest piece that works on its own; bits on one topic form a thread. |
| [D-22](agent/decisions-detail.md#d-22) | Oct 26 | Pause transcript work and build on rJokes and a simulator first: behind schedule, and less legal risk. |
| [D-23](agent/decisions-detail.md#d-23) | Oct 26 | Plan a long set bit by bit, not as one long text, so the show can change live. |
| [D-24](agent/decisions-detail.md#d-24) | Oct 26 | University GPU cluster for heavy batch jobs; everything else runs on the laptop. |
| [D-25](agent/decisions-detail.md#d-25) | Oct 27 | The director picks a style once per bit, not per joke: comedians commit to an angle, and 5 jokes give a steadier signal. |
| [D-26](agent/decisions-detail.md#d-26) | Oct 27 | The director uses LinUCB: it lost the fewest laughs in simulation. |
| [D-27](agent/decisions-detail.md#d-27) | Oct 28 | Adding "how the last bit did" as an input didn't help; keep the inputs as they are. |
| [D-28](agent/decisions-detail.md#d-28) | Oct 28 | Keep 8-bit (10-minute) shows; longer shows didn't help the director notice a mood change. |
| [D-29](agent/decisions-detail.md#d-29) | Oct 28 | Remembering tonight's results (step A) failed its test; not adopted. |
| [D-30](agent/decisions-detail.md#d-30) | Oct 31 | A per-show correction (step B) failed too: it kept repeating what worked and the room got tired. |
| [D-31](agent/decisions-detail.md#d-31) | Oct 31 | The simulated audience's style preferences come from the HumorGen paper's judge scores. |
| [D-32](agent/decisions-detail.md#d-32) | Oct 31 | rJokes is used for style statistics only, never as jokes; its safety labels proved unreliable. |
| [D-33](agent/decisions-detail.md#d-33) | Oct 31 | Docs may quote a few rJokes examples, never the dataset. |
| [D-34](agent/decisions-detail.md#d-34) | Nov 1 | Plan for comparing human (upvote) and AI judge rankings of styles, with pass criteria set before running. |
| [D-35](agent/decisions-detail.md#d-35) | Nov 1 | The human ranking compares jokes of similar length (10 length groups). |
| [D-36](agent/decisions-detail.md#d-36) | Nov 1 | Humans and the AI judge agree only weakly, so the simulator is tested with both tastes. |
| [D-37](agent/decisions-detail.md#d-37) | Nov 1 | Longer jokes get slightly more upvotes, so styles are compared only among similar-length jokes. |
| [D-38](agent/decisions-detail.md#d-38) | Nov 1 | Plan for checking whether other joke features skew the human ranking. |
| [D-39](agent/decisions-detail.md#d-39) | Nov 1 | No other feature mattered; the human ranking stands. |
| [D-40](agent/decisions-detail.md#d-40) | Nov 1 | That check's rule had a gap; an extra test confirmed the main findings. |
| [D-41](agent/decisions-detail.md#d-41) | Nov 1 | Pass criteria for testing the director under human taste. |
| [D-42](agent/decisions-detail.md#d-42) | Nov 1 | LinUCB still wins under human taste; keep it. |
| [D-43](agent/decisions-detail.md#d-43) | Nov 1 | Voice is Kokoro, run locally on the CPU: free, and no network needed at the venue. |
| [D-44](agent/decisions-detail.md#d-44) | Nov 1 | Demo moves to Dec 8; presentation and report are due Dec 14. |
| [D-45](agent/decisions-detail.md#d-45) | Nov 1 | Banter and roast replies are cut from the demo to make time. |
| [D-46](agent/decisions-detail.md#d-46) | Nov 1 | Try show-long planning now (step C1); callbacks come after the demo. |
| [D-47](agent/decisions-detail.md#d-47) | Nov 1 | Step C1 uses linear Q-learning, gets the same test as steps A and B, and runs once. |
| [D-48](agent/decisions-detail.md#d-48) | Nov 1 | C1's first run failed, likely because of a scaling mistake in its design. |
| [D-49](agent/decisions-detail.md#d-49) | Nov 1 | With the scaling fixed (C1b) it still failed; the demo keeps LinUCB. A code bug was found afterwards. |
| [D-50](agent/decisions-detail.md#d-50) | Nov 1 | With the bug fixed it still fails; LinUCB stays. |
| [D-51](agent/decisions-detail.md#d-51) | Nov 1 | rJokes may be used to test AI judges, still never as jokes. |
| [D-52](agent/decisions-detail.md#d-52) | Nov 1 | Train our own joke model (SFT, then DPO) for the demo, from premises we write. |
| [D-53](agent/decisions-detail.md#d-53) | Nov 1 | Report the true best possible score beside the "best next bit" benchmark; they differ by under 1 laugh a show. |
| [D-54](agent/decisions-detail.md#d-54) | Nov 1 | Every experiment gets a chart, or a reason it doesn't need one. |
| [D-55](agent/decisions-detail.md#d-55) | Nov 8 | The live show is one program that writes one log file per show. |
| [D-56](agent/decisions-detail.md#d-56) | Nov 8 | At each show start, the director relearns from the saved logs. |
| [D-57](agent/decisions-detail.md#d-57) | Nov 8 | The operator rates each joke's laugh with a key press (0 to 3); that is the reward. |
| [D-58](agent/decisions-detail.md#d-58) | Nov 8 | The director tries a random style 20% of the time, so the logs can test other directors later. |
| [D-59](agent/decisions-detail.md#d-59) | Nov 8 | Record show audio, only with the room's consent, to train a laugh detector later. |
| [D-60](agent/decisions-detail.md#d-60) | Nov 8 | The laugh detector rolls in step by step; key presses stay as the backup. |
| [D-61](agent/decisions-detail.md#d-61) | Nov 8 | Demo shows check the AI judge and the trained model; training on shows waits until about 30 shows. |
| [D-62](agent/decisions-detail.md#d-62) | Nov 8 | Small PRs, each reviewed by a second AI before the user sees it. |
| [D-63](agent/decisions-detail.md#d-63) | Nov 8 | Safety filter starts with word rules; add an AI check only if the rules miss. |
| [D-64](agent/decisions-detail.md#d-64) | Nov 8 | Scraps from the Loft allowed us to use its transcripts for this project, locally. |
| [D-65](agent/decisions-detail.md#d-65) | Nov 8 | Split the decisions notebook into one per project area. |
| [D-66](agent/decisions-detail.md#d-66) | Nov 8 | The hand-written joke bank gets bit structure: opener, tags, closer. |
| [D-67](agent/decisions-detail.md#d-67) | Nov 8 | Exploring 20% of the time costs under 1 laugh a show in 5 of 6 simulated rooms; the extreme room misses. |
| [D-68](agent/decisions-detail.md#d-68) | Nov 8 | Keep 20% exploration; the extreme room no longer counts against the bar. |
| [D-69](agent/decisions-detail.md#d-69) | Nov 8 | Word rules alone missed too much; add an AI check for each bit. |
| [D-70](agent/decisions-detail.md#d-70) | Nov 8 | Rules fixed and AI check added; a line is blocked whenever the check can't confirm it. |
| [D-71](agent/decisions-detail.md#d-71) | Nov 8 | The safety check may take up to 3 s (was 2 s); it runs ahead, so the audience doesn't wait. |
| [D-72](agent/decisions-detail.md#d-72) | Nov 8 | Safety bar: catch at least 75% of new, indirect offensive lines. No human pre-approves scripts. |
| [D-73](agent/decisions-detail.md#d-73) | Nov 8 | The safety filter passed on a second fresh set (24 of 25). |
| [D-74](agent/decisions-detail.md#d-74) | Nov 14 | Generator: one AI call per bit, next line prepared while one plays, filler lines between bits, hand-written jokes as backup. |
| [D-75](agent/decisions-detail.md#d-75) | Nov 14 | Backup jokes go through the word rules only, so they still work when the AI service is down. |
| [D-76](agent/decisions-detail.md#d-76) | Nov 14 | A bit keeps one storyline across its jokes, so the closer can call back to the opener. |
| [D-77](agent/decisions-detail.md#d-77) | Nov 14 | Filler lines are logged too (log format version 2). |
| [D-78](agent/decisions-detail.md#d-78) | Nov 14 | Generated shows meet the timing bars: no noticeable dead air. |
| [D-79](agent/decisions-detail.md#d-79) | Nov 15 | Room audio is recorded through the browser, only when consent is ticked for that show. |
| [D-80](agent/decisions-detail.md#d-80) | Nov 15 | Audio timing bar changed: a steady 0.24 s delay is harmless; only jitter and drift count. Both passed. |
| [D-81](agent/decisions-detail.md#d-81) | Nov 15 | Operator screen: full-width waveform, lilac theme; it never shows the chosen comedy style. |
| [D-82](agent/decisions-detail.md#d-82) | Nov 15 | The first live show uses Gemini to write jokes, so the full system gets tested. |
| [D-83](agent/decisions-detail.md#d-83) | Nov 15 | After each show the operator enters the room's 1-5 rating; a script scores the acceptance run. |
| [D-84](agent/decisions-detail.md#d-84) | Nov 15 | Review fixes to the rating prompt and the scoring script. |
| [D-85](agent/decisions-detail.md#d-85) | Nov 15 | Acceptance run passed: 3 simulated shows and 1 live show ran clean. Stage 0 is done. |
| [D-86](agent/decisions-detail.md#d-86) | Nov 24 | Two decision files: this one for people, a detailed one for agents. |
| [D-87](agent/decisions-detail.md#d-87) | Nov 24 | Docs people read stay short and plain; agent-only detail lives in `docs/agent/`. |
| [D-88](agent/decisions-detail.md#d-88) | Nov 15 | The safety check gets more time (25 s) and one retry on a garbled reply, so one slow reply no longer sends a whole bit to the backup jokes. |
| [D-89](agent/decisions-detail.md#d-89) | Nov 24 | Tag the 90k rJokes jokes with the room each suits (venue, energy, age) for Tingyu's judge; upvotes ignore the room, so a gain is not expected. |
| [D-90](agent/decisions-detail.md#d-90) | Nov 24 | A paused or cut-short show still trains the director and judge from its finished parts; pauses get log rows, and its rating isn't reported. |
| [D-91](agent/decisions-detail.md#d-91) | Nov 24 | Shows open with the best bits from earlier live shows, so the room never hears 10 s of silence. Teen rooms use only bits from teen shows. Not yet measured. |
| [D-92](agent/decisions-detail.md#d-92) | Nov 24 | If a pause hits while a joke is playing, that joke's label is left out of the reward (the rest of the bit still counts); its audio isn't used to train the laugh detector. Extends D-90. |
| [D-93](agent/decisions-detail.md#d-93) | Nov 24 | Model settings are named by the job they do (the joke writer and the safety check read their own), and the joke writer can run on Claude through the official SDK. Not run for real yet. |
| [D-94](agent/decisions-detail.md#d-94) | Nov 24 | Compare Gemini Flash-Lite, Gemini Flash and Claude Haiku 5.5 for writing bits on 60 prompts, with your blind ranking of the first 30; the bar was fixed before the run: a model must be fast, rarely time out and rarely fail, and a model with no safe thread is ranked last. |
| [D-95](agent/decisions-detail.md#d-95) | Nov 24 | A judge on Explorer (Ollama qwen3.8 27b first, Qwen 7B as fallback), taught by your votes, may rank prompts 31 to 60, but only if it agrees with your own rankings and favors no model; its added risk of a false winner is stated. |
| [D-96](agent/decisions-detail.md#d-96) | Nov 25 | After Google had an outage in the first minutes of the model comparison, the Gemini models were dropped as joke writers: the test is now Claude Haiku 5.5 against two models running on your laptop (mimo-v2.6 9b and bonsai-2 4b). The bar is the same. The cost: today's model is no longer measured, and laptop speed was measured without the voice, so it is a best case. |
| [D-97](agent/decisions-detail.md#d-97) | Nov 25 | A model that runs on the laptop only counts if Ollama keeps all of it on the graphics card, so the processor stays free for the voice. mimo-v2.6 9b fails (87% on the card); bonsai-2 4b passes. If Ollama's report is missing or looks wrong, the model does not pass. gemma4 12b is run and shown for comparison but can never be picked. The cost: four options to vote on per prompt (about 50 minutes, not 38; you accepted that), and likely only two models left in the running. |
| [D-98](agent/decisions-detail.md#d-98) | Nov 25 | Exp 19 rerun: the safety filter on Claude Haiku 5.5 catches every held-out line and rejects few clean ones, but its p95 is 4.9 s against the 3 s bar, so the filter stays on Gemini (user) |
| [D-99](agent/decisions-detail.md#d-99) | Nov 25 | Same check with Haiku's thinking turned off: still too slow (3.6 s against 3 s). Thinking was not the cause; the long reply is. The filter stays on Gemini (user) |
| [D-100](agent/decisions-detail.md#d-100) | Nov 25 | The safety filter moves to Claude Haiku with thinking off, because Gemini's free tier can fail in an outage. The speed limit moved from 3 s to 7 s (5 s target) after seeing the results; Haiku takes 3.6 s (user) |
| [D-101](agent/decisions-detail.md#d-101) | Nov 25 | Before the real run, five small local models were tried on five bits each. The test now asks the local models for the exact reply shape, as Claude is asked, so a weaker request is not held against them. mimo-v2.6 9b (which did not fit on the graphics card) is replaced by qwen3 4b-instruct, which had no failure in any of the screens. That pick is a judgment call on five bits; the 60-prompt run decides. |
| [D-102](agent/decisions-detail.md#d-102) | Nov 25 | DPO uses the judge taught by your Exp 24 votes. You rank 120 pairs of bits from the same model: 60 teach it your taste, 60 test it. It must beat a coin flip with confidence on the test pairs (at least 38 of 60). (user) |
| [D-103](agent/decisions-detail.md#d-103) | Nov 25 | In the first Exp 24 run, bonsai timed out 17 times in a row because the Ollama tray app kept restarting a server. Its block was set aside and rerun on a clean server; qwen's clean block stands. (user) |
| [D-104](agent/decisions-detail.md#d-104) | Nov 25 | Gemma 4 12B, the reference, is dropped: it runs partly on the CPU and can't finish a bit in 12 s. Haiku already shows how far the small models are from a stronger one. The ballot has 3 options again, about 38 minutes. (user) |
| [D-105](agent/decisions-detail.md#d-105) | Nov 25 | The safety rules took curly apostrophes in words like “I’ve” for names, so a quarter of qwen’s lines were wrongly rejected. Fixed; Exp 24’s safety step is redone for all models (no regeneration), and the vote starts over. (user) |
| [D-106](agent/decisions-detail.md#d-106) | Nov 25 | Only qwen writes curly quotes, long dashes and ellipses, which would give its bits away on the ballot. The ballot and the judge now show plain punctuation; the saved bits stay as written. (user) |
| [D-107](agent/decisions-detail.md#d-107) | Nov 25 | Exp 24 is done. Haiku was funnier in 16 of 30 votes, but not clearly (gap 0.5 ranks, interval touches 0). We keep bonsai-2:4b as the live writer and the training base: it is a baseline to improve, and Haiku can't be trained. Haiku is the bar SFT and DPO must reach. (user) |
| [D-108](agent/decisions-detail.md#d-108) | Nov 25 | When two models tie exactly on funniness, the faster one goes through, not the one whose name sorts first. Exp 24's result doesn't change: bonsai is faster than qwen. (user) |
| [D-109](agent/decisions-detail.md#d-109) | Nov 25 | Shows now write with bonsai on Ollama, set up exactly as Exp 24 measured it. Each show loads the model first and won't start unless it's all on the GPU, so a busy GPU can't silently turn a show into bank jokes. (user) |
| [D-110](agent/decisions-detail.md#d-110) | Nov 25 | The show now writes each bit as three separate calls, one thread of 5 jokes each, run at the same time on the same prompt. Why: one training example for DPO is then one request and the 5 jokes the audience hears. Cost: 3 calls instead of 1, so Exp 26 re-times it. Exp 24's pick stands as measured. **Exp 26 passed:** a bit takes 3.5 s at the median (5.1 s at the 95th percentile) against a 12 s budget, and no bit lost a thread. (user) |
| [D-111](agent/decisions-detail.md#d-111) | Nov 25 | Two fixes for bits that fell back to the bank: the reply format now says exactly 5 jokes per thread (on the local writer), and a badly shaped reply is asked for once more. Plus one idea to test, not yet live: a prompt that asks for one connected bit instead of five loose jokes. Exp 27 decides it; the show keeps the current prompt until then. **Exp 27 was inconclusive:** the judge preferred the new prompt, but it agreed with you on only 4 of 10 threads, which is chance, so the show keeps the current prompt (D-114 builds a better judge). |
| [D-112](agent/decisions-detail.md#d-112) | Nov 25 | Bridge lines no longer replay a line already played in any show on this laptop (the user is the only listener). Cached bridges are therefore off for your own shows, and the 10 in-character filler lines carry the waits. Turn it off with `BRIDGE_SKIP_HEARD=0` for a new audience. Checked on the logs: 17 of 17 cached bridge lines in the last show had been heard before. The filler order is shuffled each show; you have already heard all 10 filler lines, so they still repeat across shows until new ones are added (next PR). |

New decision: add its full entry to the detailed log with the next ID, then a row here in the same PR.
A test fails if the two files' IDs don't match.
