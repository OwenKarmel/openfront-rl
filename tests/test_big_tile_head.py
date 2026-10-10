"""--tile-head big: network.py's BigTileHead in place of the 1x1 conv.

  - the default (small) head is unchanged: same module names, so existing
    checkpoints still load; small and big state_dicts are not interchangeable
  - big starts uniform over legal cells (zero-initialized output conv)
  - size-invariant: (B, 1024) logits from any canvas size, like the small head
  - illegal cells masked; a boat_attack row's log_prob gets gradient into the
    big head, a non-boat row's does not
  - it actually aims: supervised on "the legal cell nearest AGENT's
    territory", it reaches high accuracy where chance is <1%

usage: python tests/test_big_tile_head.py   (exit 1 on any failure)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from envs.openfront_env import ACTIONS, MACRO_GRID
from models.network import ActorCritic

BOAT, EXPAND = ACTIONS.index("boat_attack"), ACTIONS.index("expand")
G = MACRO_GRID


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def inputs(b, size, mask):
    return (torch.randint(0, 4, (b, size, size), dtype=torch.int8), torch.rand(b, 4) * 1000,
            torch.rand(b, 1, 6) * 1000, torch.ones(b, 1, dtype=torch.bool), mask,
            torch.ones(b, 1, dtype=torch.bool), torch.ones(b, len(ACTIONS), dtype=torch.bool))


def aiming_task(b, size=256):
    """Own territory is a disc somewhere random; target = legal cell nearest it."""
    grid = torch.zeros(b, size, size, dtype=torch.int8)
    cy, cx = torch.randint(30, size - 30, (2, b))
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    grid[((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2) < 400] = 1
    mask = torch.rand(b, G, G) < 0.15
    gy, gx = torch.meshgrid(torch.arange(G) + 0.5, torch.arange(G) + 0.5, indexing="ij")
    d = (gy[None] - cy[:, None, None] * G / size) ** 2 + (gx[None] - cx[:, None, None] * G / size) ** 2
    d[~mask] = float("inf")
    x = inputs(b, size, mask)
    return (grid,) + x[1:], d.flatten(1).argmin(1)


def main() -> None:
    fails = 0
    torch.manual_seed(0)
    small, big = ActorCritic(), ActorCritic(tile_head="big")
    n_small, n_big = (sum(p.numel() for p in m.parameters()) for m in (small, big))

    print("construction")
    fails += not check("default head keeps its module (checkpoint-compatible)",
                       "tile_head_conv.weight" in small.state_dict() and not hasattr(small, "tile_head_big"))
    fails += not check("big head adds parameters", n_big > 2 * n_small, f"{n_small:,} -> {n_big:,}")
    try:
        big.load_state_dict(small.state_dict())
        refused = False
    except RuntimeError:
        refused = True
    fails += not check("a small-head checkpoint does not silently load into a big head", refused)

    print("\nforward")
    mask = torch.zeros(2, G, G, dtype=torch.bool)
    mask[0, :4, :4] = True
    mask[1, 10:20, 5] = True
    for size in (512, 256, 300):
        _, tl, _, _ = big(*inputs(2, size, mask))
        ok = tl.shape == (2, G * G)
        fails += not check(f"{size}x{size} canvas -> (2, {G * G}) logits", ok, str(tuple(tl.shape)))
    p = torch.softmax(tl, -1)
    fails += not check("illegal cells get ~0 probability", p[~mask.flatten(1)].max().item() < 1e-6)
    fails += not check("starts uniform over legal cells",
                       torch.allclose(p[0][mask[0].flatten()], torch.full((16,), 1 / 16), atol=1e-5))

    print("\ngradient")
    x = inputs(2, 128, mask)
    lp, _, _ = big.evaluate_actions(*x, torch.tensor([[BOAT, 0, 0], [EXPAND, 0, 0]]))
    big.zero_grad()
    lp[0].backward(retain_graph=True)
    boat_grad = sum(p.grad.abs().sum().item() for p in big.tile_head_big.parameters() if p.grad is not None)
    big.zero_grad()
    lp[1].backward()
    other_grad = sum(p.grad.abs().sum().item() for p in big.tile_head_big.parameters() if p.grad is not None)
    fails += not check("boat row's log_prob trains the big head", boat_grad > 0, f"{boat_grad:.3g}")
    fails += not check("non-boat row's does not", other_grad == 0)

    print("\naiming (supervised, 600 steps)")
    # GPU if there is one; otherwise a few threads, not every core -- this
    # usually runs beside live training.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    net = ActorCritic(tile_head="big").to(dev)
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    for _ in range(600):
        x, y = aiming_task(32)
        loss = F.cross_entropy(net(*(t.to(dev) for t in x))[1], y.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y = aiming_task(500)
        acc = (net(*(t.to(dev) for t in x))[1].argmax(1).cpu() == y).float().mean().item()
    fails += not check("finds the legal cell nearest AGENT's territory", acc > 0.6,
                       f"accuracy {acc:.2f}, chance ~{1 / (0.15 * G * G):.3f}")

    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
