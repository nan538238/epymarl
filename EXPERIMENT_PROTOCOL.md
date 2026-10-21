# Fast feedback screening protocol

This is the original T1 LBF mechanism screen, not UAV validation or a claim of a new state-machine algorithm.

Task: 8x8, three level-1 agents, two level-2 foods, full observation, 100-step horizon. Original team reward is summed once and replicated for training; original_return is the primary metric. One fixed simulated rule binds helper 1 and leader 0 to the lexicographically first food coordinate. It activates when leader Manhattan distance <=3. All methods stop shaping after target collection. No feedback is used during evaluation. Training and evaluation random streams are separated. A target remains bound until episode end.

Budget: initially 100000 environment steps per method, seeds 11 and 22; feedback starts at 20000 steps. Four workers, matched IPPO config, strength 0.1, per-episode absolute shaping cap 0.5. This deliberately shorter screen is not the proposal's full 1M-step experiment. Negative results are budget-specific, not proof of impossibility.

First run no-feedback seeds. If both final success rates >=0.95, stop this configuration for ceiling effects. If both final original returns <=0.05, inspect exploration before spending on all comparisons. Otherwise run persistent and context at the same budget/seeds. Compare final raw team return, success and learning curves. A condition-gating lead is provisional only if paired differences have the same positive direction and success improvement >=0.05 in both seeds. Positive screens require weak (quarter-strength persistent) and time-decay controls before further investment. These are engineering gates, not statistical significance tests.

Measure condition fraction and switches independently of the manager. If natural condition changes are scarce, the task is not evidence for advice staleness. Direct condition and context modes are mathematically equivalent under this no-hysteresis design and are tested for reward-sequence equivalence; duplicate training does not create an independent novelty claim.

Use 200 held-out evaluation episodes (seeds starting at 200000) on fixed-budget policies for final decisions; training-time evaluation starts at 100000. No test-set tuning or best-checkpoint selection. Keep all outcomes, including failures.
