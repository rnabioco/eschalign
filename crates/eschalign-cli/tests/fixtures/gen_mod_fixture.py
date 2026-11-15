#!/usr/bin/env python3
"""Regenerate the --mod-tiebreak fixture: mod_panel.fa, mod_reads.fq, mod_paralogs.toml.

Synthetic, fixed seed, no real data. Two paralog pairs, Thr-CGT-1/2 and
Val-GAC-1/2, each pair identical but for two bases (POS). A read carries, at
those two bases, a base found in *neither* paralog, so it mismatches both
twice and the pair ties on the canonical score. The sidecar declares sites at
those two bases on one paralog of each pair only (the "true" one), expecting
the read's base: Thr-CGT-2 and Val-GAC-1. Reads are named `<true paralog>.<n>`.
Run from this directory:

    python3 gen_mod_fixture.py
"""
import random

rng = random.Random(20261115)
POS = (30, 45)  # 0-based, interior of a 76 nt reference
N_READS = 6


def rand(n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def pair(a):
    b = list(a)
    for p in POS:
        b[p] = {"A": "C", "C": "G", "G": "T", "T": "A"}[a[p]]
    return a, "".join(b)


thr1, thr2 = pair(rand(76))
val1, val2 = pair(rand(76))
panel = [("Thr-CGT-1", thr1), ("Thr-CGT-2", thr2), ("Val-GAC-1", val1), ("Val-GAC-2", val2)]
truth = {"Thr-CGT-2": (thr1, thr2), "Val-GAC-1": (val1, val2)}

with open("mod_panel.fa", "w") as f:
    for n, s in panel:
        f.write(f">{n}\n{s}\n")

sites = []
with open("mod_reads.fq", "w") as f:
    for name, (a, b) in truth.items():
        true = a if name.endswith("-1") else b
        subs = []
        for p in POS:
            choices = [c for c in "ACGT" if c not in (a[p], b[p])]
            subs.append(rng.choice(choices))
        for n in range(N_READS):
            read = list(true)
            # The same substitutions in every read: the declared site expects them.
            for p, c in zip(POS, subs):
                read[p] = c
            f.write(f"@{name}.{n}\n{''.join(read)}\n+\n{'I' * len(read)}\n")
        for p, c in zip(POS, subs):
            sites.append((name, p + 1, c))

with open("mod_paralogs.toml", "w") as f:
    for name, pos, c in sites:
        f.write(
            f'[[site]]\nref = "{name}"\npos = {pos}\nmod = "synthetic"\n'
            f'expect = ["mismatch:{c}"]\nwindow = [0, 0]\n\n'
        )
