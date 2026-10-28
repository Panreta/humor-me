# sim: simulated audience

The primary RL evaluation for M1 (design.md, "Simulated audience environment"). Plain numpy,
gym-style `reset`/`step`, thousands of episodes per second.

```python
from sim.env import AudienceEnv, Context
env = AudienceEnv(shift_at=20)                     # optional mid-set preference shift
obs = env.reset(Context("bar", "normal", "mixed"), seed=0)
obs, reward, done, info = env.step(1)              # action = index into env.modes
info["level"], info["regret"]                      # 0-3 laugh level; oracle gap for this joke
```

| Piece | Model |
|---|---|
| Latent | `base[mode] + venue[mode] + age[mode] + energy + novelty·e^(−t/τ) − fatigue·uses_in_window + shift` |
| Laugh level | latent + N(0, noise), counted against thresholds `[0, 1, 2]` → 0–3 |
| Reward | level / 3 (samplers work on [0, 1]; logs keep the level) |
| Oracle | `expected_rewards()`: exact E[level]/3 per mode for the next joke; regret = best − chosen |

All numbers live in [priors.yaml](priors.yaml). Only `base` has a source (HumorGen judge win
rates); venue/age/energy offsets are made up and the report must say so. PR 3 swaps `base`
for rJokes-derived values.

Not modeled yet: callbacks (delayed reward), switching penalty (that's a policy-side cost).
