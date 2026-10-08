#!/usr/bin/env python3
"""DDTree core: build a draft tree from a block-diffusion drafter's per-position
marginals, its tree-attention mask + depth positions, and the accept walk.

Pure/deterministic — NO GPU, NO model. This is the algorithmic heart of the
frontier acceptance lever (arxiv 2604.12989). The only remaining piece to wire it
into MLX/native is the batched target forward with (mask, positions) from here.

Reusable by:
  - MLX DFlash fork (verify tree in one mx target forward with tree mask)
  - gemma-metal native step_verify `.wip` (swap causal mask -> tree_mask, positions)
"""
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TreeNode:
    token: int          # token id at this node
    parent: int         # index into nodes[] of parent (-1 for root)
    depth: int          # position depth within the block (root = 0)
    logprob: float      # cumulative log-prob of the prefix ending here


def build_tree(per_pos_logprobs: List[List[Tuple[int, float]]],
               root_token: int,
               budget: int,
               top_k: int) -> List[TreeNode]:
    """DDTree best-first enumeration (paper Algorithm 1).

    per_pos_logprobs[d] = sorted (token, logprob) list for block depth d (0-based),
       descending by logprob, already truncated to >= top_k entries.
    root_token = the last committed token (tree root, depth 0, attends to KV ctx).
    budget = max number of *drafted* nodes (excludes root).
    top_k = max branching factor per depth.

    Returns nodes[0]=root, then up to `budget` drafted nodes in descending
    cumulative-prefix-logprob order. parent indices point earlier in the list,
    so a single left-to-right pass is a valid topological order.
    """
    L = len(per_pos_logprobs)
    root = TreeNode(token=root_token, parent=-1, depth=0, logprob=0.0)
    nodes: List[TreeNode] = [root]
    if L == 0 or budget <= 0:
        return nodes

    # Heap entries: (-cum_logprob, tie, depth, rank, parent_node_idx)
    #   depth  = block position this candidate fills (0-based -> node depth d+1)
    #   rank   = index into per_pos_logprobs[depth] (which top-k token)
    # Expansions from a popped node:
    #   sibling: same depth, rank+1 (alternative token at this position)
    #   child:   depth+1, rank 0    (extend the accepted prefix by one position)
    def tok_lp(depth, rank):
        return per_pos_logprobs[depth][rank]

    tie = 0
    t0, lp0 = tok_lp(0, 0)
    heap = [(-lp0, tie, 0, 0, 0)]  # first child of root
    while heap and (len(nodes) - 1) < budget:
        neg_cum, _, depth, rank, parent_idx = heapq.heappop(heap)
        cum = -neg_cum
        token, _ = tok_lp(depth, rank)
        node_idx = len(nodes)
        nodes.append(TreeNode(token=token, parent=parent_idx, depth=depth + 1, logprob=cum))

        # sibling: alternative token at the SAME depth, sharing this node's parent
        if rank + 1 < min(top_k, len(per_pos_logprobs[depth])):
            _, sib_lp = tok_lp(depth, rank + 1)
            parent_cum = nodes[parent_idx].logprob
            tie += 1
            heapq.heappush(heap, (-(parent_cum + sib_lp), tie, depth, rank + 1, parent_idx))
        # child: extend this node to the next depth, rank 0
        if depth + 1 < L:
            _, ch_lp = tok_lp(depth + 1, 0)
            tie += 1
            heapq.heappush(heap, (-(cum + ch_lp), tie, depth + 1, 0, node_idx))
    return nodes


def tree_attention_mask(nodes: List[TreeNode]) -> List[List[bool]]:
    """[N][N] boolean: mask[j][i] True iff query node j may attend to key node i,
    i.e. i is j itself or an ancestor of j (root included). Context KV (before the
    tree) is always attendable and handled separately by the caller."""
    n = len(nodes)
    anc = [set() for _ in range(n)]
    for j in range(n):
        cur = j
        while cur != -1:
            anc[j].add(cur)
            cur = nodes[cur].parent
    return [[i in anc[j] for i in range(n)] for j in range(n)]


def tree_positions(nodes: List[TreeNode], ctx_offset: int) -> List[int]:
    """Per-node RoPE position = ctx_offset + depth. Siblings share a position
    (that is exactly what makes tree verify correct)."""
    return [ctx_offset + nd.depth for nd in nodes]


def accept_walk(nodes: List[TreeNode], target_argmax: List[int]) -> Tuple[List[int], int]:
    """Greedy tree accept. target_argmax[i] = argmax of the target's logits at the
    position of node i (i.e. the token the target would emit AFTER node i).

    Walk root->deepest: from the current node, accept a child whose token equals
    the target's argmax at the current node. Returns (accepted_tokens, bonus_token)
    where accepted_tokens are the committed draft tokens (excluding root) and
    bonus_token is the target's argmax at the last accepted node (always correct,
    even at zero accept)."""
    children = {}
    for idx, nd in enumerate(nodes):
        children.setdefault(nd.parent, []).append(idx)

    cur = 0  # root
    accepted: List[int] = []
    while True:
        want = target_argmax[cur]
        nxt = None
        for c in children.get(cur, []):
            if nodes[c].token == want:
                nxt = c
                break
        if nxt is None:
            break
        accepted.append(nodes[nxt].token)
        cur = nxt
    bonus = target_argmax[cur]
    return accepted, bonus
