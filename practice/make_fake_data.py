"""Make a FAKE match in the exact format of the real hackathon data, to practise on.

The real match data is released on October 20. Until then, this writes a simulated match into
data/tactical/ with the same file names, the same file formats and the same timeline as the real
one, so everything the organisers provide runs on it unchanged:

    python starter/baseline.py                 the baseline
    starter/03_first_submission_FAKE.ipynb     notebook 03, scored against the fake answers
    python evaluation/run_evaluation.py --gt evaluation/gt/FAKE_possession_gt.csv

NOTHING HERE IS REAL FOOTBALL. It is a simple simulation (see fake_football.py): 22 players in a
4-4-2, passes, interceptions, tackles, throw-ins, goal kicks and free kicks, with the tracking made
noisy on purpose -- the ball vanishes about 40% of the time, like in the real data.

Files written:
    data/tactical/BAR-ATH_tactical_atd.xml            tracking metadata (teams, track slots)
    data/tactical/BAR-ATH_tactical_atd.txt            tracking, 199,319 frames, players and ball
    data/tactical/BAR-ATH_tactical_smart_tagging.xml  set-piece tags (a cut-down Smart Tagging)
    data/tactical/FAKE_DATA_README.txt                a reminder that this is fake
    evaluation/gt/FAKE_possession_gt.csv               the true answer for every frame

WHEN THE REAL DATA ARRIVES: unzip it into data/ and let it overwrite these files (or delete
data/tactical first). Do not submit anything built on the fake match.

    python practice/make_fake_data.py              # seed 1
    python practice/make_fake_data.py --seed 7     # a different fake match
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from fake_football import Match, PITCH_L, PITCH_W, FPS, H, A  # noqa: E402
from games import asset_paths  # noqa: E402

N_FRAMES = 199319
# The same timeline as the real match (see DATA.md, "Which frames are graded").
HALF1 = (11185, 90651)          # graded, kickoff at 11185
HALF2 = (110808, 190197)        # graded, kickoff at 113742 (before that: players lining up, D)
KICKOFF2 = 113742
TEAM_NAMES = ("Barcelona", "Atletic Club")     # spelled like the real Smart Tagging does
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "FAKE_possession_gt.csv")


def simulate_half(seed, n_frames, kick_team, mirror):
    """Play one half. mirror=True flips the pitch so the teams swap ends after halftime."""
    match = Match(seed, n_frames=n_frames, kick_team=kick_team)
    match.players[kick_team][10] = [52.5, 34.0]          # kicker on the centre spot
    match.ball = np.array([52.5, 34.0])
    match.play()
    match.add_tracking_noise()
    ball, players = match.ball_track, match.player_track
    if mirror:
        ball[:, 0] = PITCH_L - ball[:, 0]
        players[..., 0] = PITCH_L - players[..., 0]
    return ball, players, match.label, match.events


def lining_up(rng, n_frames, mirror):
    """Players standing around in formation before the 2nd-half kickoff. Ball dead."""
    from fake_football import FORMATION
    home = FORMATION.copy()
    away = np.column_stack([PITCH_L - FORMATION[:, 0], FORMATION[:, 1]])
    base = np.stack([home, away])
    base[:, :, 0] = np.clip((base[:, :, 0] - 52.5) * 0.5 + 52.5, 1, PITCH_L - 1)
    if mirror:
        base[..., 0] = PITCH_L - base[..., 0]
    drift = np.cumsum(rng.normal(0, 0.03, (n_frames, 2, 11, 2)), axis=0)
    players = base[None] + drift + rng.normal(0, 0.25, (n_frames, 2, 11, 2))
    ball = np.full((n_frames, 2), np.nan)
    show = rng.random(n_frames) < 0.3
    ball[show] = [52.5, 34.0]
    return ball, players


def build(seed):
    rng = np.random.default_rng(seed + 1000)
    ball = np.full((N_FRAMES, 2), np.nan)
    players = np.full((N_FRAMES, 2, 11, 2), np.nan)
    label = np.full(N_FRAMES, "X", dtype="<U1")
    events = []      # (video_frame, type, team_index or None, half)

    print("  simulating 1st half ...")
    a, b = HALF1
    hb, hp, hl, he = simulate_half(seed, b - a + 1, H, mirror=False)
    ball[a:b + 1], players[a:b + 1], label[a:b + 1] = hb, hp, hl
    events += [(a + f, kind, team, "1st Half") for f, kind, team in he]

    print("  simulating 2nd half ...")
    a, b = HALF2
    lb, lp = lining_up(rng, KICKOFF2 - a, mirror=True)
    ball[a:KICKOFF2], players[a:KICKOFF2], label[a:KICKOFF2] = lb, lp, "D"
    hb, hp, hl, he = simulate_half(seed + 1, b - KICKOFF2 + 1, A, mirror=True)
    ball[KICKOFF2:b + 1], players[KICKOFF2:b + 1], label[KICKOFF2:b + 1] = hb, hp, hl
    events += [(KICKOFF2 + f, kind, team, "2nd Half") for f, kind, team in he]

    # outside the halves (before, halftime, after) almost nobody is tracked -- like the real file
    return ball, players, label, events


def speeds(xy):
    """m/s from frame-to-frame movement, NaN where the object is not seen."""
    # Smooth the positions first (about a third of a second), like a real tracker does --
    # otherwise the frame-to-frame jitter alone looks like sprinting.
    import pandas as pd
    flat = xy.reshape(len(xy), -1)
    smooth = pd.DataFrame(flat).rolling(9, center=True, min_periods=3).mean().to_numpy()
    smooth = smooth.reshape(xy.shape)
    v = np.full(xy.shape[:-1], np.nan)
    v[1:] = np.linalg.norm(np.diff(smooth, axis=0), axis=-1) * FPS
    v[np.isnan(xy[..., 0])] = np.nan
    return v


def fmt(values):
    """Format an array of floats as strings, 'NaN' for missing -- the ATD text convention."""
    out = np.char.mod("%.5f", np.nan_to_num(values, nan=0.0))
    return np.where(np.isnan(values), "NaN", out)


def write_atd_txt(path, ball, players):
    """Raw ATD: '<frame_id>:x,y,s;x,y,s;...;x,y,s:x,y,s' with frame_id counting from 1."""
    n = len(ball)
    slots = players.reshape(n, 22, 2)                     # H1..H11 then A1..A11
    slot_speed = speeds(slots)
    ball_speed = speeds(ball)
    px, py = fmt(slots[..., 0] / PITCH_L), fmt(slots[..., 1] / PITCH_W)
    ps = np.where(np.isnan(slots[..., 0]), "NaN", fmt(np.round(slot_speed, 2)))
    bx, by = fmt(ball[:, 0] / PITCH_L), fmt(ball[:, 1] / PITCH_W)
    bs = np.where(np.isnan(ball[:, 0]), "NaN", fmt(np.round(ball_speed, 2)))
    with open(path, "w", newline="\n") as handle:
        for f in range(n):
            groups = ";".join(f"{px[f, s]},{py[f, s]},{ps[f, s]}" for s in range(22))
            handle.write(f"{f + 1}:{groups}:{bx[f]},{by[f]},{bs[f]}\n")


def write_atd_xml(path, n_frames):
    """FIFA/EPTS metadata, Metrica style. 22 track slots, 11 per team."""
    players, channels, refs = [], [], []
    for slot in range(22):
        team_id = "FIFATMA" if slot < 11 else "FIFATMB"
        pid = f"P{slot + 1:04d}"
        players.append(
            f'      <Player id="{pid}" teamId="{team_id}">\n'
            f'        <Name>Track {slot + 1}</Name>\n'
            f'        <ShirtNumber>{slot + 1}</ShirtNumber>\n'
            f'      </Player>')
        for ch in ("x", "y", "speed"):
            channels.append(f'      <PlayerChannel id="player{slot + 1}_{ch}" playerId="{pid}" channelId="{ch}"/>')
        refs.append('        <SplitRegister separator=",">\n' + "\n".join(
            f'          <PlayerChannelRef playerChannelId="player{slot + 1}_{ch}"/>'
            for ch in ("x", "y", "speed")) + '\n        </SplitRegister>')

    def param(name, value):
        return f"        <ProviderParameter><Name>{name}</Name><Value>{value}</Value></ProviderParameter>"

    first = (HALF1[0] + 1, HALF1[1] + 1)          # ATD frame ids count from 1
    second = (HALF2[0] + 1, HALF2[1] + 1)
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<main>
  <Metadata>
    <GlobalConfig>
      <FileDate>2026-10-09T00:00:00.000Z</FileDate>
      <TrackingType>Optical</TrackingType>
      <ProviderName>FAKE - practice data, not real</ProviderName>
      <FrameRate>{FPS}</FrameRate>
      <ProviderGlobalParameters>
{param("first_half_start", first[0])}
{param("first_half_end", first[1])}
{param("second_half_start", second[0])}
{param("second_half_end", second[1])}
{param("first_extra_half_start", "")}
{param("first_extra_half_end", "")}
{param("second_extra_half_start", "")}
{param("second_extra_half_end", "")}
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions>
      <Session id="Session_1">
        <SessionType>Match</SessionType>
        <MatchParameters>
          <Score idLocalTeam="FIFATMA" idVisitingTeam="FIFATMB">
            <LocalTeamScore>0</LocalTeamScore>
            <VisitingTeamScore>0</VisitingTeamScore>
          </Score>
        </MatchParameters>
        <FieldSize>
          <Width>{int(PITCH_L)}</Width>
          <Height>{int(PITCH_W)}</Height>
        </FieldSize>
      </Session>
    </Sessions>
    <Teams>
      <Team id="FIFATMA">
        <Name>FAKE {TEAM_NAMES[0]}</Name>
        <ProviderTeamsParameters>
{param("attack_direction_first_half", "left_to_right")}
        </ProviderTeamsParameters>
      </Team>
      <Team id="FIFATMB">
        <Name>FAKE Athletic Club</Name>
        <ProviderTeamsParameters>
{param("attack_direction_first_half", "right_to_left")}
        </ProviderTeamsParameters>
      </Team>
    </Teams>
    <Players>
{chr(10).join(players)}
    </Players>
    <Devices>
      <Device id="device1">
        <Name>Device 1</Name>
        <Sensors>
          <Sensor id="position">
            <Name>Position</Name>
            <Channels>
              <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
              <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
            </Channels>
          </Sensor>
          <Sensor id="speed">
            <Name>Speed</Name>
            <Channels>
              <Channel id="speed"><Name>speed</Name><Unit>meters per second</Unit></Channel>
            </Channels>
          </Sensor>
        </Sensors>
      </Device>
    </Devices>
    <PlayerChannels>
{chr(10).join(channels)}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification separator=":" startFrame="1" endFrame="{n_frames}">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{chr(10).join(refs)}
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
        <BallChannelRef channelId="speed"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
"""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(xml)


def write_smart_tagging(path, events, rng):
    """SportsCode XML with SET PIECES instances, timed in seconds from the first video frame.

    Like the real Smart Tagging it is automatic and imperfect: about 1 in 10 restarts is not
    tagged, the windows are a little loose, and throw-ins have Team = N/A.
    """
    instances = []
    for i, (frame, kind, team, half) in enumerate(events):
        if kind == "Kick Off":
            kind = "Kick Off (Start)"
        elif rng.random() < 0.10:
            continue
        team_text = "N/A" if team == "N/A" else TEAM_NAMES[0 if team == "Home" else 1]
        start = (frame - rng.uniform(25, 75)) / FPS
        end = (frame + rng.uniform(50, 125)) / FPS
        instances.append(f"""    <instance>
      <ID>{len(instances) + 1}</ID>
      <start>{start:.2f}</start>
      <end>{end:.2f}</end>
      <code>SET PIECES</code>
      <label><group>Half</group><text>{half}</text></label>
      <label><group>Team</group><text>{team_text}</text></label>
      <label><group>Type</group><text>{kind}</text></label>
    </instance>""")
    xml = ('<?xml version="1.0" encoding="utf-8"?>\n<file>\n  <ALL_INSTANCES>\n'
           + "\n".join(instances) + "\n  </ALL_INSTANCES>\n</file>\n")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(xml)
    return len(instances)


def main():
    parser = argparse.ArgumentParser(description="Write a fake match in the real data format.")
    parser.add_argument("--seed", type=int, default=1, help="change it for a different match")
    args = parser.parse_args()
    t0 = time.time()

    paths = asset_paths()
    folder = os.path.dirname(paths["atd_raw"])
    os.makedirs(folder, exist_ok=True)

    print(f"FAKE match, seed {args.seed}")
    ball, players, label, events = build(args.seed)

    print("  writing tracking (about 70 MB, takes a minute) ...")
    write_atd_xml(paths["atd_meta"], N_FRAMES)
    write_atd_txt(paths["atd_raw"], ball, players)
    n_tags = write_smart_tagging(paths["smart_tagging"], events,
                                 np.random.default_rng(args.seed + 2000))
    with open(GT_PATH, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{f},{l}\n" for f, l in enumerate(label))
    n_segments = int(np.sum((label[1:] != label[:-1]) & (label[1:] != "X"))) + int(label[0] != "X")
    alive_l = np.isin(label, ["H", "A"])
    n_poss = int(np.sum(alive_l[1:] & (label[1:] != label[:-1]))) + int(alive_l[0])
    meta = {"game": "FAKE", "fps": FPS, "n_video_frames": N_FRAMES,
            "n_graded_frames": int(np.sum(label != "X")), "n_possessions": n_poss,
            "n_segments": n_segments,
            "mean_possession_duration_s": round(float(alive_l.sum()) / n_poss / FPS, 3),
            "note": f"FAKE practice answers from practice/make_fake_data.py, seed {args.seed}"}
    import json
    with open(GT_PATH[:-len(".csv")] + "_meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(folder, "FAKE_DATA_README.txt"), "w") as handle:
        handle.write("These files are FAKE practice data made by practice/make_fake_data.py "
                     f"(seed {args.seed}).\nThey are not the real match. When the real data "
                     "arrives, delete this folder and unzip the real files here.\n")

    graded = label != "X"
    alive = np.isin(label, ["H", "A"])
    spells = np.sum(alive[1:] & (label[1:] != label[:-1])) + int(alive[0])
    seen = ~np.isnan(ball[:, 0])
    print(f"\n  graded frames  {graded.sum()}   H {np.sum(label == 'H')}  A {np.sum(label == 'A')}"
          f"  D {np.sum(label == 'D')}")
    print(f"  possessions    {spells}, mean {alive.sum() / spells / FPS:.1f} s")
    print(f"  ball visible   {100 * seen[graded].mean():.1f}% of graded frames")
    print(f"  set-piece tags {n_tags}")
    print(f"\n  wrote {folder}")
    print(f"  wrote {os.path.relpath(GT_PATH, ROOT)}   <- the answers, score against this")
    print(f"  done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
