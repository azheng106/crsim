"""Self-play training for MiniClash.

Two independent LinearQAgents (one per team) play each other. Every tick, each agent
observes from its own perspective, picks an action, and does a TD(0) update from the
per-team reward the engine returns. Because both sides improve at once, each agent's
opponent is non-stationary -- the defining feature (and challenge) of self-play.

Usage:
    python train.py                 # train, then save crsim_agents.npz
    python train.py --episodes 800
    python train.py --watch         # load saved agents and watch them in pygame
"""

import argparse

import numpy as np

from ClashSim import MiniClash, P0, P1
from agent import LinearQAgent, encode, legal_mask, action_to_env


def run_episode(env: MiniClash, agents: dict[int, LinearQAgent],
                max_ticks: int, learn: bool = True) -> int | None:
    """Play one game; agents learn online if `learn`. Returns the winner (or None)."""
    env.reset()
    done = False
    ticks = 0

    while not done and ticks < max_ticks:
        phi = {t: encode(env, t) for t in (P0, P1)}
        mask = {t: legal_mask(env, t) for t in (P0, P1)}
        acts = {t: agents[t].select_action(phi[t], mask[t], greedy=not learn)
                for t in (P0, P1)}

        env_actions = {t: action_to_env(t, acts[t]) for t in (P0, P1)}
        _, rewards, done = env.step(env_actions)
        ticks += 1

        if learn:
            phi_next = {t: encode(env, t) for t in (P0, P1)}
            mask_next = {t: legal_mask(env, t) for t in (P0, P1)}
            for t in (P0, P1):
                agents[t].update(phi[t], acts[t], rewards[t],
                                 phi_next[t], mask_next[t], done)

    return env.state.winner


def train(episodes: int, max_ticks: int, out_path: str) -> dict[int, LinearQAgent]:
    env = MiniClash()
    agents = {
        P0: LinearQAgent(seed=0),
        P1: LinearQAgent(seed=1),
    }

    eps_start, eps_end = 1.0, 0.05
    window = []  # recent winners for a rolling report

    for ep in range(episodes):
        # Linear epsilon decay over the whole run.
        frac = ep / max(1, episodes - 1)
        eps = eps_start + (eps_end - eps_start) * frac
        for a in agents.values():
            a.eps = eps

        winner = run_episode(env, agents, max_ticks, learn=True)
        window.append(winner)

        report_every = max(1, episodes // 20)
        if (ep + 1) % report_every == 0:
            w0 = window.count(P0)
            w1 = window.count(P1)
            draws = len(window) - w0 - w1
            n = len(window)
            print(f"ep {ep + 1:5d}/{episodes} | eps={eps:.2f} | "
                  f"last {n}: P0 {w0/n:5.1%}  P1 {w1/n:5.1%}  draw {draws/n:5.1%}")
            window = []

    agents[P0].save(out_path.replace(".npz", "_p0.npz"))
    agents[P1].save(out_path.replace(".npz", "_p1.npz"))
    print(f"Saved agents to {out_path.replace('.npz', '_p0.npz')} / _p1.npz")
    return agents


def evaluate(agents: dict[int, LinearQAgent], games: int, max_ticks: int) -> None:
    """Greedy head-to-head so we measure learned skill, not exploration noise."""
    env = MiniClash()
    for a in agents.values():
        a.eps = 0.0
    tally = {P0: 0, P1: 0, None: 0}
    for _ in range(games):
        tally[run_episode(env, agents, max_ticks, learn=False)] += 1
    print(f"Greedy eval over {games}: "
          f"P0 {tally[P0]}  P1 {tally[P1]}  draw {tally[None]}")


def watch(model_path: str, max_ticks: int) -> None:
    """Load trained agents and render a live match."""
    import pygame
    from ClashSim import Renderer

    agents = {
        P0: LinearQAgent.load(model_path.replace(".npz", "_p0.npz")),
        P1: LinearQAgent.load(model_path.replace(".npz", "_p1.npz")),
    }
    for a in agents.values():
        a.eps = 0.0

    env = MiniClash()
    ren = Renderer()
    running, paused, ticks = True, False, 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    env.reset(); ticks = 0

        if not paused and env.state.winner is None and ticks < max_ticks:
            phi = {t: encode(env, t) for t in (P0, P1)}
            mask = {t: legal_mask(env, t) for t in (P0, P1)}
            env_actions = {t: action_to_env(t, agents[t].select_action(phi[t], mask[t], greedy=True))
                           for t in (P0, P1)}
            env.step(env_actions)
            ticks += 1

        ren.draw(env.state)
        ren.tick()
    ren.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--model", default="crsim_agents.npz")
    ap.add_argument("--watch", action="store_true", help="render saved agents instead of training")
    ap.add_argument("--eval", type=int, default=100, help="greedy eval games after training")
    args = ap.parse_args()

    if args.watch:
        watch(args.model, args.max_ticks)
        return

    agents = train(args.episodes, args.max_ticks, args.model)
    if args.eval > 0:
        evaluate(agents, args.eval, args.max_ticks)


if __name__ == "__main__":
    main()
