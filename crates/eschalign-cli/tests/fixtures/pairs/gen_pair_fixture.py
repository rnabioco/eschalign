#!/usr/bin/env python3
"""Regenerate the paired-end fixture: pairs_panel.fa, pairs_R1.fq, pairs_R2.fq.

Everything is synthesized from a fixed seed (no real data). The truth set is
built by hand, one pair per behaviour the paired-end tests pin; `align_paired.rs`
states what each pair must produce. Run from this directory:

    python3 gen_pair_fixture.py
"""
import random

R1_ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
R2_ADAPTER = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"

rng = random.Random(20261004)


def rand(n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def rc(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def other(b):
    return {"A": "C", "C": "G", "G": "T", "T": "A"}[b]


# Two paralogs differing at one base (position 100), and two unrelated references.
p1 = rand(150)
p2 = p1[:100] + other(p1[100]) + p1[101:]
u1 = rand(150)
u2 = rand(150)
panel = {"P1": p1, "P2": p2, "U1": u1, "U2": u2}

pairs = []


def add(name, r1, r2):
    pairs.append((name, r1, r2))


# Mate 1 lies where P1 and P2 are identical (alone it ties); mate 2 covers
# position 100, which only P1 explains.
add("paralog", p1[20:80], rc(p1[90:150]))
# Both mates in the identical region: the pair ties between P1 and P2.
add("pair_tie", p1[0:50], rc(p1[40:90]))
# Insert of 40 nt, both mates read 20 nt into their adapter.
tpl = u1[30:70]
add("short_insert", tpl + R1_ADAPTER[:20], rc(tpl) + R2_ADAPTER[:20])
# Only mate 1 reads through (15 adapter bases); mate 2 is shorter and stays inside.
add("one_mate_reads_through", u2[30:70] + R1_ADAPTER[:15], rc(u2[35:70]))
# A mismatch on the last base of mate 1 and on the first base of mate 2.
m1 = u1[10:70]
m1 = m1[:-1] + other(m1[-1])
m2 = rc(u1[70:130])
m2 = other(m2[0]) + m2[1:]
add("terminal_mismatch", m1, m2)
# Insert of 150: proper by default, not under --max-insert 120.
add("over_max_insert", u1[0:50], rc(u1[100:150]))
# Both mates read as sequenced on the same strand (FF).
add("wrong_orientation", u2[0:50], u2[100:150])
# Mate 2 is junk.
add("one_unmapped", u2[20:80], rand(60))
# An ordinary pair.
add("plain", u2[10:70], rc(u2[80:140]))
# Discordant: mate 1 comes from U1, mate 2 from U2 (each an unambiguous best).
add("discordant", u1[10:70], rc(u2[80:140]))


def qual(n, shift):
    return "".join(chr(33 + 20 + (i * 3 + shift) % 20) for i in range(n))


with open("pairs_panel.fa", "w") as f:
    for k, v in panel.items():
        f.write(f">{k}\n{v}\n")
with open("pairs_R1.fq", "w") as f1, open("pairs_R2.fq", "w") as f2:
    for name, a, b in pairs:
        f1.write(f"@{name}\n{a}\n+\n{qual(len(a), 0)}\n")
        f2.write(f"@{name}\n{b}\n+\n{qual(len(b), 7)}\n")
