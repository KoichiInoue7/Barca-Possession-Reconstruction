"""Make a FAKE football match to practise on while the real data is not out yet.

Nothing here is real. It is a simple simulation: two teams of 11 move around in a rough
formation, one player carries the ball, it gets passed, sometimes intercepted or tackled, and
sometimes it goes out of play or the referee stops play for a foul. Then, to make it feel like
the real automatic tracking, we add noise and make the ball disappear about 40% of the time.

It writes three files into practice/fake_match/:

    tracking.csv   one row per frame: ball_x, ball_y, then H1_x, H1_y ... A11_x, A11_y
                   positions are 0-1 (like the real data). An empty cell means "not seen".
    answers.csv    frame,label -- the TRUE answer for every frame (H, A or D)
    events.csv     a few automatic "tags" for restarts (throw-ins, goal kicks, free kicks),
                   a bit like the real Smart Tagging. Throw-ins have team N/A, like the real one.

Run it:
    python practice/make_fake_match.py               # the default fake match (seed 1)
    python practice/make_fake_match.py --seed 7      # a different fake match

Use a second seed to check your method works on a match you did not tune it on.
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---- settings you can play with ------------------------------------------------------------
FPS = 25
MINUTES = 15                    # length of the fake match
PITCH_L, PITCH_W = 105.0, 68.0  # metres
BALL_MISSING_TARGET = 0.40      # about how often the ball is invisible during play
POSITION_NOISE = 0.25           # metres of jitter added to every tracked position
# --------------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "fake_match")

H, A = 0, 1
TEAM_LETTER = {H: "H", A: "A"}

# A 4-4-2 for the home team attacking to the right (+x). Away is the mirror image.
FORMATION = np.array([
    [5, 34],                                   # goalkeeper
    [22, 10], [20, 26], [20, 42], [22, 58],    # defenders
    [40, 10], [38, 27], [38, 41], [40, 58],    # midfielders
    [55, 28], [55, 40],                        # forwards
], dtype=float)


class Match:
    def __init__(self, seed, n_frames=MINUTES * 60 * FPS, kick_team=0):
        self.rng = np.random.default_rng(seed)
        self.kick_team = kick_team
        n = n_frames
        self.n = n
        self.players = np.zeros((2, 11, 2))
        self.players[H] = FORMATION
        self.players[A] = np.column_stack([PITCH_L - FORMATION[:, 0], FORMATION[:, 1]])
        self.wander = np.zeros((2, 11, 2))      # slow random drift around the formation

        # what we record
        self.ball_track = np.full((n, 2), np.nan)
        self.player_track = np.full((n, 2, 11, 2), np.nan)
        self.label = np.empty(n, dtype="<U1")
        self.events = []

        # match state
        self.ball = np.array([52.5, 34.0])
        self.team = H                 # team in possession (or last in possession)
        self.carrier = 10             # index of the player carrying the ball
        self.mode = "carry"           # carry | flight | contest | dead
        self.timer = 0
        self.flight = None
        self.dead = None

    # ---------------------------------------------------------------- helpers
    def attack_dir(self, team):
        return 1.0 if team == H else -1.0

    def nearest(self, team, point):
        d = np.linalg.norm(self.players[team] - point, axis=1)
        return int(np.argmin(d)), float(np.min(d))

    def start_carry(self, team, player):
        self.mode = "carry"
        self.team, self.carrier = team, player
        self.timer = int(self.rng.uniform(0.6, 3.0) * FPS)

    def start_dead(self, kind, restart_team, spot, stop_ball_at=None):
        """Ball out of play. Restart after a while by restart_team from spot."""
        seconds = {"Throw-in": (3, 8), "Goal Kick": (6, 13), "Free Kick": (5, 11)}[kind]
        self.mode = "dead"
        self.dead = {
            "kind": kind, "team": restart_team, "spot": np.array(spot, dtype=float),
            "left": int(self.rng.uniform(*seconds) * FPS),
            "ball_to": np.array(stop_ball_at if stop_ball_at is not None else spot, dtype=float),
        }

    # ---------------------------------------------------------------- one frame of football
    def step_players(self):
        rng = self.rng
        bx, by = self.ball
        for team in (H, A):
            att = self.attack_dir(team)
            base = FORMATION.copy() if team == H else np.column_stack(
                [PITCH_L - FORMATION[:, 0], FORMATION[:, 1]])
            # the whole team slides with the ball; attackers push up, defenders drop
            shift_x = 0.45 * (bx - 52.5) + (6 * att if team == self.team else -4 * att)
            shift_y = 0.30 * (by - 34.0)
            target = base + np.array([shift_x, shift_y])
            target[0] = base[0] + np.array([0.15 * (bx - 52.5), 0.2 * shift_y])  # keeper stays home
            self.wander[team] += rng.normal(0, 0.08, (11, 2)) - 0.01 * self.wander[team]
            target = target + self.wander[team]

            # the two nearest opponents of the ball carrier press the ball
            if self.mode in ("carry", "contest") and team != self.team:
                d = np.linalg.norm(self.players[team] - self.ball, axis=1)
                for i in np.argsort(d)[:2]:
                    target[i] = self.ball + rng.normal(0, 1.2, 2)
            # the carrier runs with the ball
            if self.mode == "carry" and team == self.team:
                target[self.carrier] = self.players[team][self.carrier] + np.array([att * 1.5, rng.normal(0, 0.6)])
            # the receiver of a pass runs to meet it
            if self.mode == "flight" and team == self.flight["catch_team"]:
                target[self.flight["catcher"]] = self.flight["to"]
            # during a stoppage the restart taker walks to the ball
            if self.mode == "dead" and team == self.dead["team"]:
                taker = self.dead.setdefault("taker", self.nearest(team, self.dead["spot"])[0])
                target[taker] = self.dead["spot"]

            speed = 0.05 if self.mode == "dead" else 0.09
            step = speed * (target - self.players[team])
            step_len = np.linalg.norm(step, axis=1, keepdims=True)
            max_step = 8.0 / FPS                          # nobody runs faster than 8 m/s
            step = np.where(step_len > max_step, step / np.maximum(step_len, 1e-9) * max_step, step)
            self.players[team] += step
            self.players[team][:, 0] = np.clip(self.players[team][:, 0], 0.5, PITCH_L - 0.5)
            self.players[team][:, 1] = np.clip(self.players[team][:, 1], 0.5, PITCH_W - 0.5)

    def step_ball(self, f):
        rng = self.rng
        if self.mode == "carry":
            p = self.players[self.team][self.carrier]
            self.ball = p + np.array([self.attack_dir(self.team) * 0.8, 0.0])
            self.label[f] = TEAM_LETTER[self.team]
            self.timer -= 1
            if self.timer <= 0:
                self.choose_action()

        elif self.mode == "flight":
            fl = self.flight
            fl["t"] += 1
            k = min(1.0, fl["t"] / fl["frames"])
            self.ball = fl["from"] + k * (fl["to"] - fl["from"])
            self.label[f] = TEAM_LETTER[self.team]        # still the passer's team until control
            if fl["t"] >= fl["frames"]:
                if fl["outcome"] == "out":
                    self.start_dead(fl["dead_kind"], 1 - self.team, fl["restart_spot"])
                else:
                    self.start_carry(fl["catch_team"], fl["catcher"])

        elif self.mode == "contest":
            # a tackle: the ball bounces around between two players for a moment.
            # possession only changes when the tackler actually controls it.
            c = self.contest
            c["left"] -= 1
            mid = 0.5 * (self.players[self.team][self.carrier] + self.players[c["team"]][c["player"]])
            self.ball = mid + rng.normal(0, 0.8, 2)
            self.label[f] = TEAM_LETTER[self.team]
            if c["left"] <= 0:
                if c["wins"]:
                    self.start_carry(c["team"], c["player"])
                else:
                    self.start_carry(self.team, self.carrier)

        elif self.mode == "dead":
            dd = self.dead
            dd["left"] -= 1
            # the ball drifts to the restart spot (ball boy, player picking it up...)
            self.ball = self.ball + 0.06 * (dd["spot"] - self.ball)
            self.label[f] = "D"
            if dd["left"] <= 0:
                self.ball = dd["spot"].copy()
                team_tag = "N/A" if dd["kind"] == "Throw-in" else (
                    "Home" if dd["team"] == H else "Away")
                self.events.append((f, dd["kind"], team_tag))
                taker = dd.get("taker", self.nearest(dd["team"], dd["spot"])[0])
                self.players[dd["team"]][taker] = dd["spot"]
                self.team, self.carrier = dd["team"], taker
                self.pass_ball(force_complete=True)
                self.label[f] = TEAM_LETTER[self.team]

    def choose_action(self):
        rng = self.rng
        att = self.attack_dir(self.team)
        x = self.ball[0]
        in_final_third = (x > 70) if self.team == H else (x < 35)
        r = rng.random()
        if r < 0.07:                                       # tackled
            opp, dist = self.nearest(1 - self.team, self.ball)
            self.mode = "contest"
            self.contest = {"team": 1 - self.team, "player": opp,
                            "left": int(rng.uniform(0.3, 1.2) * FPS),
                            "wins": rng.random() < 0.6}
        elif r < 0.09:                                     # foul -> free kick, same team keeps it
            self.start_dead("Free Kick", self.team, self.ball.copy())
        elif in_final_third and r < 0.15:                  # shot that misses -> goal kick
            goal_x = PITCH_L + 2 if self.team == H else -2
            gk_spot = [PITCH_L - 5.5, 34 + rng.normal(0, 5)] if self.team == H else [5.5, 34 + rng.normal(0, 5)]
            self.flight = {"from": self.ball.copy(), "to": np.array([goal_x, 34 + rng.normal(0, 6)]),
                           "t": 0, "frames": int(0.8 * FPS), "outcome": "out",
                           "dead_kind": "Goal Kick", "restart_spot": gk_spot,
                           "catch_team": self.team, "catcher": self.carrier}
            self.mode = "flight"
        else:
            self.pass_ball()

    def pass_ball(self, force_complete=False):
        rng = self.rng
        team = self.team
        att = self.attack_dir(team)
        mates = [i for i in range(11) if i != self.carrier]
        d = np.linalg.norm(self.players[team][mates] - self.ball, axis=1)
        forward = (self.players[team][mates][:, 0] - self.ball[0]) * att
        score = -np.abs(d - 18) + 0.4 * forward + rng.normal(0, 6, len(mates))
        catcher = mates[int(np.argmax(score))]
        to = self.players[team][catcher] + np.array([att * rng.uniform(0, 4), rng.normal(0, 2)])
        dist = np.linalg.norm(to - self.ball)
        frames = max(6, int(dist / rng.uniform(12, 20) * FPS))
        outcome, catch_team = "complete", team
        r = rng.random()
        if not force_complete and r < 0.09:                # intercepted
            catch_team = 1 - team
            catcher, _ = self.nearest(catch_team, self.ball + 0.6 * (to - self.ball))
            to = self.ball + rng.uniform(0.4, 0.8) * (to - self.ball)
            frames = max(5, int(frames * 0.6))
        elif not force_complete and r < 0.13:              # overhit, goes out for a throw-in
            outcome = "out"
            side = 0.0 if to[1] < 34 else PITCH_W
            to = np.array([np.clip(to[0], 2, PITCH_L - 2), side + (-3 if side == 0 else 3)])
        self.flight = {"from": self.ball.copy(), "to": to, "t": 0, "frames": frames,
                       "outcome": outcome, "catch_team": catch_team, "catcher": catcher}
        if outcome == "out":
            self.flight["dead_kind"] = "Throw-in"
            self.flight["restart_spot"] = [to[0], 0.0 if to[1] < 34 else PITCH_W]
        self.mode = "flight"

    # ---------------------------------------------------------------- run it
    def play(self):
        # kick off
        self.events.append((0, "Kick Off", "Home" if self.kick_team == H else "Away"))
        self.start_carry(self.kick_team, 10)
        for f in range(self.n):
            self.step_players()
            self.step_ball(f)
            self.ball_track[f] = self.ball
            self.player_track[f] = self.players
        return self

    def add_tracking_noise(self):
        """Make it look like automatic tracking: jitter, gaps, and a ball that keeps vanishing."""
        rng = self.rng
        n = self.n
        self.ball_track += rng.normal(0, POSITION_NOISE, self.ball_track.shape)
        self.player_track += rng.normal(0, POSITION_NOISE, self.player_track.shape)

        # The ball disappears in bursts. It is lost more easily when it is in the air
        # (passes), in crowds, and while play is stopped.
        crowd = np.zeros(n)
        for f in range(0, n):
            d = np.linalg.norm(self.player_track[f].reshape(22, 2) - self.ball_track[f], axis=1)
            crowd[f] = np.sum(d < 4)
        dead = self.label == "D"
        visible = np.ones(n, dtype=bool)
        f, seen = 0, True
        while f < n:
            if seen:
                mean_run = 75 / (1 + 0.30 * crowd[f]) * (0.4 if dead[f] else 1.0)
            else:
                mean_run = 30 * (2.0 if dead[f] else 1.0)
            run = max(1, int(rng.exponential(mean_run)))
            visible[f:f + run] = seen
            f += run
            seen = not seen
        self.ball_track[~visible] = np.nan

        # each player is lost for a short while now and then
        for team in (H, A):
            for p in range(11):
                for _ in range(rng.poisson(self.n / (60 * FPS) * 1.5)):
                    start = rng.integers(0, n)
                    self.player_track[start:start + rng.integers(5, 60), team, p] = np.nan
        return visible

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        cols = {"frame": np.arange(self.n),
                "ball_x": self.ball_track[:, 0] / PITCH_L,
                "ball_y": self.ball_track[:, 1] / PITCH_W}
        for team, letter in ((H, "H"), (A, "A")):
            for p in range(11):
                cols[f"{letter}{p + 1}_x"] = self.player_track[:, team, p, 0] / PITCH_L
                cols[f"{letter}{p + 1}_y"] = self.player_track[:, team, p, 1] / PITCH_W
        tracking = pd.DataFrame(cols)
        tracking.to_csv(os.path.join(out_dir, "tracking.csv"), index=False, float_format="%.4f")
        pd.DataFrame({"frame": np.arange(self.n), "label": self.label}).to_csv(
            os.path.join(out_dir, "answers.csv"), index=False)
        pd.DataFrame(self.events, columns=["frame", "type", "team"]).to_csv(
            os.path.join(out_dir, "events.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Make a fake match to practise on.")
    parser.add_argument("--seed", type=int, default=1, help="change it to get a different match")
    parser.add_argument("--out", default=OUT_DIR, help="folder to write into")
    args = parser.parse_args()

    match = Match(args.seed).play()
    visible = match.add_tracking_noise()
    match.save(args.out)

    labels = match.label
    alive = labels != "D"
    poss = np.sum((labels[1:] != labels[:-1]) & (labels[1:] != "D")) + (labels[0] != "D")
    print(f"fake match, seed {args.seed}: {match.n} frames ({MINUTES} minutes at {FPS} fps)")
    print(f"  labels   H {np.sum(labels == 'H')}  A {np.sum(labels == 'A')}  D {np.sum(labels == 'D')}")
    print(f"  possessions {poss}, mean {np.sum(alive) / max(poss, 1) / FPS:.1f} s")
    print(f"  ball missing on {100 * np.mean(~visible[alive]):.0f}% of in-play frames")
    print(f"  wrote tracking.csv, answers.csv, events.csv to {args.out}")


if __name__ == "__main__":
    main()
