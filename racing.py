"""Step 1: one self-learning car on an oval track (top-down 2D).

Track = a closed centerline (list of x,y points) + a width. To use the real
Hungaroring later, just replace make_oval_centerline() with points traced
from a map. Car has a name/team so multi-car racing is easy to add later.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

W, H = 900, 600


def make_oval_centerline(n=200, a=330, b=190):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([W / 2 + a * np.cos(t), H / 2 + b * np.sin(t)], axis=1)


class Track:
    def __init__(self, centerline, width=90):
        self.c = np.asarray(centerline, dtype=np.float64)
        self.n = len(self.c)
        self.width = width
        seg = np.roll(self.c, -1, axis=0) - self.c
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.length = self.seg_len.sum()
        # direction of travel at each centerline point
        self.dir = seg / self.seg_len[:, None]

    def nearest(self, pts):
        """pts: (m,2) -> (index of nearest centerline point, distance) per point."""
        d = np.linalg.norm(pts[:, None, :] - self.c[None, :, :], axis=2)
        idx = d.argmin(axis=1)
        return idx, d[np.arange(len(pts)), idx]

    def on_track(self, pts):
        return self.nearest(pts)[1] < self.width / 2

    def edges(self):
        normal = np.stack([-self.dir[:, 1], self.dir[:, 0]], axis=1)
        return (self.c + normal * self.width / 2, self.c - normal * self.width / 2)


class Car:
    def __init__(self, name="Driver", team="Team", color=(220, 30, 30)):
        self.name, self.team, self.color = name, team, color
        self.x = self.y = self.heading = self.speed = 0.0


class RaceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    N_RAYS, RAY_LEN, RAY_STEP = 7, 160.0, 8.0
    MAX_SPEED, ACCEL, BRAKE, MAX_STEER = 9.0, 0.25, 0.45, 0.08

    def __init__(self, render_mode=None, track=None):
        self.track = track or Track(make_oval_centerline())
        self.car = Car("Max", "Red Bull")
        self.render_mode = render_mode
        self.ray_angles = np.linspace(-math.pi / 2, math.pi / 2, self.N_RAYS)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # steer, throttle/brake
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.N_RAYS + 1,), dtype=np.float32)
        self._screen = None

    # ---- sensing ----
    def _rays(self):
        car, steps = self.car, np.arange(1, int(self.RAY_LEN / self.RAY_STEP) + 1) * self.RAY_STEP
        ang = car.heading + self.ray_angles
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)                      # (R,2)
        pts = np.array([car.x, car.y]) + d[:, None, :] * steps[None, :, None]  # (R,S,2)
        ok = self.track.on_track(pts.reshape(-1, 2)).reshape(len(ang), len(steps))
        first_off = np.where(ok.all(axis=1), len(steps), (~ok).argmax(axis=1))
        return (first_off * self.RAY_STEP / self.RAY_LEN).clip(0, 1)

    def _obs(self):
        return np.append(self._rays(), self.car.speed / self.MAX_SPEED).astype(np.float32)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        i = int(self.np_random.integers(self.track.n))
        c = self.car
        c.x, c.y = self.track.c[i]
        dx, dy = self.track.dir[i]
        c.heading, c.speed = math.atan2(dy, dx), 0.0
        self.prev_idx, self.steps = i, 0
        return self._obs(), {}

    def step(self, action):
        steer, thr = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
        c = self.car
        c.heading += steer * self.MAX_STEER * (0.3 + c.speed / self.MAX_SPEED)
        c.speed = float(np.clip(c.speed + (thr * self.ACCEL if thr > 0 else thr * self.BRAKE), 0, self.MAX_SPEED))
        c.x += math.cos(c.heading) * c.speed
        c.y += math.sin(c.heading) * c.speed
        self.steps += 1

        idx, dist = (v[0] for v in self.track.nearest(np.array([[c.x, c.y]])))
        delta = (idx - self.prev_idx + self.track.n // 2) % self.track.n - self.track.n // 2  # wrapped
        self.prev_idx = idx
        reward = delta * (self.track.length / self.track.n) * 0.1   # progress along the track

        crashed = dist >= self.track.width / 2
        if crashed:
            reward = -10.0
        truncated = self.steps >= 2000
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, bool(crashed), truncated, {}

    # ---- drawing ----
    def render(self):
        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((W, H))
            self._font = pygame.font.SysFont("arial", 14)
            self._clock = pygame.time.Clock()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
        s = self._screen
        s.fill((40, 120, 50))
        outer, inner = self.track.edges()
        pygame.draw.polygon(s, (70, 70, 75), outer.tolist())
        pygame.draw.polygon(s, (40, 120, 50), inner.tolist())
        pygame.draw.polygon(s, (240, 240, 240), outer.tolist(), 3)
        pygame.draw.polygon(s, (240, 240, 240), inner.tolist(), 3)
        c = self.car
        fwd = np.array([math.cos(c.heading), math.sin(c.heading)])
        side = np.array([-fwd[1], fwd[0]])
        p = np.array([c.x, c.y])
        body = [p + fwd * 12 + side * 5, p + fwd * 12 - side * 5, p - fwd * 12 - side * 5, p - fwd * 12 + side * 5]
        pygame.draw.polygon(s, c.color, [b.tolist() for b in body])
        s.blit(self._font.render(f"{c.name} ({c.team})", True, (255, 255, 255)), (c.x + 14, c.y - 18))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
