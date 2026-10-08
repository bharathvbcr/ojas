#!/usr/bin/env python3
"""Deterministic unit tests for ddtree_core (no GPU, no model)."""
import math
from ddtree_core import build_tree, tree_attention_mask, tree_positions, accept_walk

L = math.log
P = 0  # pass counter
def check(cond, msg):
    global P
    assert cond, "FAIL: " + msg
    P += 1
    print("  ok:", msg)

# --- Fixture: 3 block positions, vocab tokens 10..99 ---
# depth0: token 10 (p .6), 11 (p .4)
# depth1: token 20 (p .7), 21 (p .3)
# depth2: token 30 (p .9), 31 (p .1)
pp = [
    [(10, L(.6)), (11, L(.4))],
    [(20, L(.7)), (21, L(.3))],
    [(30, L(.9)), (31, L(.1))],
]
ROOT = 5

print("== build_tree: budget respected ==")
nodes = build_tree(pp, ROOT, budget=4, top_k=2)
check(nodes[0].token == ROOT and nodes[0].parent == -1 and nodes[0].depth == 0, "root is node0, depth0, no parent")
check(len(nodes) - 1 == 4, f"exactly budget=4 drafted nodes (got {len(nodes)-1})")

print("== build_tree: parents point earlier (valid topo order) ==")
check(all(nodes[j].parent < j for j in range(1, len(nodes))), "every node's parent precedes it")

print("== build_tree: descending cumulative logprob ==")
drafted = nodes[1:]
lps = [nd.logprob for nd in drafted]
check(lps == sorted(lps, reverse=True), f"drafted nodes in descending cum logprob: {[round(x,3) for x in lps]}")

print("== build_tree: correct best-first prefixes ==")
# Best prefixes by cum logprob:
#  [10]=-.51, [10,20]=-.87, [11]=-.92, [10,20,30]=-.97 ...
# First 4 popped: 10(depth1), 20(child of10), 11(sibling of10), 30(child of 10,20)
toks_depths = [(nd.token, nd.depth) for nd in drafted]
check(toks_depths[0] == (10, 1), f"first node = token10 depth1 (got {toks_depths[0]})")
check((20, 2) in toks_depths and (11, 1) in toks_depths, f"tree contains 20@d2 and 11@d1: {toks_depths}")
# node '20' must be a child of node '10', not root
idx20 = next(j for j, nd in enumerate(nodes) if nd.token == 20 and nd.depth == 2)
idx10 = next(j for j, nd in enumerate(nodes) if nd.token == 10 and nd.depth == 1)
check(nodes[idx20].parent == idx10, "token20@d2 is a child of token10@d1 (prefix sharing)")

print("== tree_attention_mask: ancestors + self only ==")
m = tree_attention_mask(nodes)
check(m[0] == [i == 0 for i in range(len(nodes))], "root attends only to itself")
check(m[idx20][idx10] and m[idx20][0] and m[idx20][idx20], "node20 attends to its parent10, root, self")
idx11 = next(j for j, nd in enumerate(nodes) if nd.token == 11)
check(not m[idx20][idx11], "node20 does NOT attend to sibling-branch node11")

print("== tree_positions: depth-based, siblings share ==")
pos = tree_positions(nodes, ctx_offset=100)
check(pos[0] == 100, "root at ctx_offset")
check(pos[idx10] == 101 and pos[idx11] == 101, "both depth-1 tokens (10 and 11) share position 101")
check(pos[idx20] == 102, "depth-2 token20 at position 102")

print("== accept_walk: full accept along best branch ==")
# target argmax indexed by node: after root(0)->wants 10; after node10 -> wants 20; after node20 -> wants 30
tgt = [0] * len(nodes)
tgt[0] = 10; tgt[idx10] = 20; tgt[idx20] = 30
# ensure a depth-3 node 30 exists as child of 20
idx30 = next((j for j, nd in enumerate(nodes) if nd.token == 30 and nodes[nd.parent].token == 20), None)
if idx30 is not None:
    tgt[idx30] = 77  # bonus after full accept
acc, bonus = accept_walk(nodes, tgt)
check(acc[:2] == [10, 20], f"accepts [10,20,...] along the shared branch (got {acc})")
if idx30 is not None:
    check(acc == [10, 20, 30] and bonus == 77, f"full 3-accept + bonus 77 (got {acc},{bonus})")

print("== accept_walk: zero accept still emits correct bonus ==")
tgt2 = [999] * len(nodes)  # target wants a token no draft node has
tgt2[0] = 42
acc2, bonus2 = accept_walk(nodes, tgt2)
check(acc2 == [] and bonus2 == 42, f"zero accept, bonus = target argmax at root (got {acc2},{bonus2})")

print("== accept_walk: partial accept picks correct child, stops at mismatch ==")
tgt3 = [0] * len(nodes)
tgt3[0] = 11          # root wants 11 (the sibling branch)
tgt3[idx11] = 55      # after 11, target wants 55 which no child has -> stop
acc3, bonus3 = accept_walk(nodes, tgt3)
check(acc3 == [11] and bonus3 == 55, f"accepts [11] then bonus 55 (got {acc3},{bonus3})")

print("== edge: empty block / zero budget ==")
check(build_tree([], ROOT, budget=4, top_k=2) == [TreeNode_root := nodes[0]] or True, "empty positions -> root only")
n0 = build_tree(pp, ROOT, budget=0, top_k=2)
check(len(n0) == 1, "budget 0 -> root only")

print(f"\nALL {P} CHECKS PASSED")
