"""Dagelijkse token-budget: 50 tokens/dag, kosten schalen met modelcomplexiteit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DAILY_LIMIT = 50

# Hoe lastiger het model, hoe meer tokens
SHAPE_COST = {
    "box": 3,
    "sphere": 4,
    "cylinder": 4,
    "cone": 4,
    "torus": 5,
    "pyramid": 3,
    "sword": 8,
    "house": 10,
    "tree": 11,
    "tower": 19,
    "crystal": 12,
    "robot": 50,
    "castle": 60,
}

DIFFICULTY_LABEL = {
    range(1, 6): "Eenvoudig",
    range(6, 12): "Gemiddeld",
    range(12, 17): "Lastig",
    range(17, 100): "Expert",
}


def difficulty_for_cost(cost: int) -> str:
    for r, label in DIFFICULTY_LABEL.items():
        if cost in r:
            return label
    return "Expert"


@dataclass
class TokenState:
    day: str
    remaining: int
    used_today: int


class TokenWallet:
    """Persistente dagelijkse wallet (50 tokens per kalenderdag)."""

    def __init__(self, path: str | Path | None = None, daily_limit: int = DAILY_LIMIT):
        if path is None:
            from .paths import app_dir

            path = app_dir() / ".neotorch_tokens.json"
        self.path = Path(path)
        self.daily_limit = daily_limit
        self.state = self._load()

    def _today(self) -> str:
        return date.today().isoformat()

    def _load(self) -> TokenState:
        today = self._today()
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("day") == today:
                    rem = int(data.get("remaining", self.daily_limit))
                    used = int(data.get("used_today", self.daily_limit - rem))
                    return TokenState(day=today, remaining=max(0, rem), used_today=max(0, used))
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        return TokenState(day=today, remaining=self.daily_limit, used_today=0)

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "day": self.state.day,
                    "remaining": self.state.remaining,
                    "used_today": self.state.used_today,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def refresh_if_new_day(self) -> None:
        today = self._today()
        if self.state.day != today:
            self.state = TokenState(day=today, remaining=self.daily_limit, used_today=0)
            self.save()

    @property
    def remaining(self) -> int:
        self.refresh_if_new_day()
        return self.state.remaining

    @property
    def used_today(self) -> int:
        self.refresh_if_new_day()
        return self.state.used_today

    @property
    def limit(self) -> int:
        return self.daily_limit

    def can_afford(self, cost: int) -> bool:
        return self.remaining >= cost

    def spend(self, cost: int) -> bool:
        self.refresh_if_new_day()
        if cost < 0 or self.state.remaining < cost:
            return False
        self.state.remaining -= cost
        self.state.used_today += cost
        self.save()
        return True
