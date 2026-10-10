# Practice with a FAKE match

The real match data comes out on **October 20**. Until then, this folder makes a **fake match**
in the exact same format, so the organisers' own code (baseline, notebooks, scorer) runs on it.

It is a simulation, not real football: 22 players, passes, interceptions, tackles, throw-ins,
goal kicks and free kicks. The tracking is made messy on purpose, like the real data. The ball
is invisible about 44% of the time and players drop out now and then.

## 1. Make the fake match (once, ~30 seconds)

Run from the project folder (`more-than-a-hack-2027-main`):

```
python practice/make_fake_data.py
```

That writes:

| file | what it is |
|---|---|
| `data/tactical/BAR-ATH_tactical_atd.xml` | tracking info: which track slot is on which team |
| `data/tactical/BAR-ATH_tactical_atd.txt` | the tracking: player and ball positions, every frame |
| `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | automatic tags for restarts (throw-ins, goal kicks, ...) |
| `evaluation/gt/FAKE_possession_gt.csv` | **the answers**: H / A / D for every frame |

There's no video or homography. You don't need them to start.

## 2. Run the baseline and score it

```
python starter/baseline.py --out evaluation/results_fake
python evaluation/run_evaluation.py --gt evaluation/gt/FAKE_possession_gt.csv --results evaluation/results_fake --out evaluation/fake_scores
```

The `--out evaluation/results_fake` part keeps fake results apart from the real ones in
`evaluation/results`.

## 3. Use the notebook

Open `starter/03_first_submission_FAKE.ipynb`. It's the organisers' notebook 03, changed only
to score against the fake answers. Run the cells top to bottom with **Shift+Enter**.

Notebooks 01 and 02 should also run on the fake files, since they read the same file names.

## 4. Try your own ideas

Write your own version of the "who has the ball" code and score it the same way. A second fake
match is a good test that your idea isn't just lucky on one match:

```
python practice/make_fake_data.py --seed 7
```

## ⚠️ When the real data arrives

1. **Delete `data/tactical`** (it holds the fake files), then unzip the real data into `data/`.
2. Never submit anything made from the fake match.
3. `evaluation/gt/FAKE_possession_gt.csv` and `evaluation/results_fake` can be deleted too.

## How the fake match differs from the real one

- 22 track slots that each follow one player. The real data has 46 slots that swap and break up.
- Simple, made-up football. Real players, passes and deflections are much messier.
- Only set-piece tags. The real Smart Tagging also has phases like Build Up and Progression.
- No video, no homography, no broadcast camera.

So a method that works here is a good start, not a guarantee.
