# -*- coding: utf-8 -*-
"""Verify the mathematician's corrections and the decisive new results.

V1 SYMMETRY. My paper claims E[U|p] "moves 0.863 -> 0.980 as p goes 0.3 -> 0.7".
   Claimed WRONG: when h+l=1, U(k)=U(N-k) so E[U|0.3] == E[U|0.7] EXACTLY, and 0.980 is p=0.5.
V2 NO-ESCAPE-BY-NORMALISATION. Plug-in Fleiss/ICC-style normalisation s^2/(phat(1-phat)) is
   claimed to equal N/(N-1) IDENTICALLY for 0<k<N - i.e. the standard fix is maximally degenerate.
V3 A_90 IS ALSO DEGENERATE. 50% crossing near p=0.153, flat above p=0.6.
V4 INVERSION BOUND. G([0,t]) >= Phi_A(t) - Abar: turns the critique into a positive capability claim.
V5 NULL OVER-PRODUCES dU. A model with NO reliability parameter predicts a LARGER relative rise in
   U than the paper reports.
"""
import numpy as np
from math import comb

N, HI, LO = 10, 90, 10


def U_of_k(k, N=N, hi=HI, lo=LO):
    x = sorted([1.0] * k + [0.0] * (N - k))
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def A_of_k(k, N=N, hi=HI):
    x = sorted([1.0] * k + [0.0] * (N - k))
    return float(np.percentile(x, hi))


def E(p, fn, N=N):
    return sum(comb(N, k) * p ** k * (1 - p) ** (N - k) * fn(k, N) for k in range(N + 1))


print("=" * 74)
print("V1 SYMMETRY: is E[U|0.3] == E[U|0.7] exactly?")
print("=" * 74)
e3, e5, e7 = E(0.3, U_of_k), E(0.5, U_of_k), E(0.7, U_of_k)
print(f"  E[U|0.3] = {e3:.9f}")
print(f"  E[U|0.5] = {e5:.9f}   <- the maximum")
print(f"  E[U|0.7] = {e7:.9f}")
print(f"  |E[0.3]-E[0.7]| = {abs(e3-e7):.2e}  -> symmetry {'CONFIRMED' if abs(e3-e7)<1e-12 else 'FAILS'}")
print(f"  U(k)==U(N-k) for all k: {all(abs(U_of_k(k)-U_of_k(N-k))<1e-12 for k in range(N+1))}")
print("  >>> MY PAPER'S SENTENCE IS WRONG: E[U|p] is NON-MONOTONE, peaking at p=0.5.")

print()
print("=" * 74)
print("V2 NO ESCAPE: does base-rate normalisation (Fleiss/ICC style) fix it?")
print("=" * 74)
print("   k    s^2 (unbiased)   phat(1-phat)     ratio")
for k in range(1, N):
    phat = k / N
    s2 = k * (N - k) / (N * (N - 1))          # unbiased sample variance of 0/1 data
    print(f"  {k:2d}   {s2:.6f}         {phat*(1-phat):.6f}      {s2/(phat*(1-phat)):.6f}")
print(f"  >>> ratio is IDENTICALLY N/(N-1) = {N/(N-1):.6f} for every 0<k<N.")
print("  >>> The standard normalisation is a CONSTANT: it destroys all information.")
print("  >>> This kills the obvious reviewer objection 'why not just use Fleiss kappa'.")

print()
print("=" * 74)
print("V3 A_90 IS ALSO DEGENERATE (not a capability measure)")
print("=" * 74)
for p in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.153):
    print(f"   p={p:<6.3f} E[A_90|p] = {E(p, A_of_k):.6f}")
lo_, hi_ = 0.0, 1.0
for _ in range(60):
    mid = (lo_ + hi_) / 2
    if E(mid, A_of_k) < 0.5:
        lo_ = mid
    else:
        hi_ = mid
print(f"  >>> A_90 crosses 0.5 at p = {(lo_+hi_)/2:.4f}, and is flat (>0.998) above p=0.6.")
print("  >>> So A and U are two THRESHOLD READOUTS of the same difficulty distribution.")

print()
print("=" * 74)
print("V4 INVERSION BOUND: G([0,t]) >= Phi_A(t) - Abar")
print("=" * 74)
print("  Their Abar_sharded <= 0.84 (16% relative drop from at most 1.0).")
for t in (0.3, 0.5, 0.7):
    print(f"   t={t}:  Phi_A(t)={E(t, A_of_k):.6f}  ->  G([0,{t}]) >= {E(t, A_of_k)-0.84:+.3f}")
print("  >>> At least ~15% of instructions fall below p=0.5 multi-turn.")
print("  >>> That is a CAPABILITY COLLAPSE, provable from their own aptitude statistic.")

print()
print("=" * 74)
print("V5 NULL WITH NO RELIABILITY PARAMETER vs their reported +112% / -16%")
print("=" * 74)
rng = np.random.default_rng(0)
print(f"  {'conc':>6}{'dA %':>9}{'dU %':>9}")
for c in (2, 5, 15, 30):
    out = []
    for mean in (0.90, 0.65):
        a, b = mean * c, (1 - mean) * c
        ps = rng.beta(a, b, 4000)
        A = np.mean([E(p, A_of_k) for p in ps])
        Uv = np.mean([E(p, U_of_k) for p in ps])
        out.append((A, Uv))
    (A0, U0), (A1, U1) = out
    print(f"  {c:>6}{(A1/A0-1)*100:>8.1f}%{(U1/U0-1)*100:>8.1f}%")
print("  paper:   -16.0%   +112.0%")
print("  >>> The null OVER-produces the unreliability rise and UNDER-produces the aptitude drop.")
print("  >>> The anomalous component is the CAPABILITY COLLAPSE, not the variance.")
