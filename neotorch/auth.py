"""Lokale NeoTorch-accounts: register / login / sessie (voor Ultra-tab)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


_USER_RE = re.compile(r"^[a-zA-Z0-9_\-.]{3,24}$")
_ITERATIONS = 120_000


@dataclass
class AuthUser:
    username: str


class AuthStore:
    """Persistente lokale users + optionele sessie naast de .exe / projectroot."""

    def __init__(self, path: str | Path | None = None, session_path: str | Path | None = None):
        from .paths import app_dir

        root = app_dir()
        self.path = Path(path) if path else root / ".neotorch_auth.json"
        self.session_path = Path(session_path) if session_path else root / ".neotorch_session.json"
        self._users: Dict[str, dict] = self._load_users()
        self.user: Optional[str] = self._load_session()

    @property
    def logged_in(self) -> bool:
        return bool(self.user)

    def _load_users(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            users = data.get("users", {})
            if isinstance(users, dict):
                return {str(k).lower(): v for k, v in users.items() if isinstance(v, dict)}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
        return {}

    def _save_users(self) -> None:
        self.path.write_text(
            json.dumps({"users": self._users}, indent=2),
            encoding="utf-8",
        )

    def _load_session(self) -> Optional[str]:
        if not self.session_path.exists():
            return None
        try:
            data = json.loads(self.session_path.read_text(encoding="utf-8"))
            name = str(data.get("user", "")).lower().strip()
            if name and name in self._users:
                return name
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
        return None

    def _save_session(self, username: Optional[str]) -> None:
        if not username:
            if self.session_path.exists():
                try:
                    self.session_path.unlink()
                except OSError:
                    pass
            return
        self.session_path.write_text(
            json.dumps({"user": username}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _hash(password: str, salt: bytes) -> str:
        dig = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
        return dig.hex()

    def register(self, username: str, password: str) -> Tuple[bool, str]:
        name = username.strip().lower()
        if not _USER_RE.match(name):
            return False, "Gebruikersnaam: 3–24 tekens (letters, cijfers, _.-)"
        if len(password) < 4:
            return False, "Wachtwoord minstens 4 tekens."
        if name in self._users:
            return False, "Deze gebruikersnaam bestaat al."
        salt = secrets.token_bytes(16)
        self._users[name] = {
            "salt": salt.hex(),
            "hash": self._hash(password, salt),
            "iter": _ITERATIONS,
        }
        self._save_users()
        self.user = name
        self._save_session(name)
        return True, f"Account aangemaakt — welkom, {name}."

    def login(self, username: str, password: str) -> Tuple[bool, str]:
        name = username.strip().lower()
        rec = self._users.get(name)
        if not rec:
            return False, "Onbekende gebruikersnaam of wachtwoord."
        try:
            salt = bytes.fromhex(str(rec["salt"]))
            expected = str(rec["hash"])
            iters = int(rec.get("iter", _ITERATIONS))
        except (KeyError, TypeError, ValueError):
            return False, "Accountbestand corrupt."
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters).hex()
        if not hmac.compare_digest(got, expected):
            return False, "Onbekende gebruikersnaam of wachtwoord."
        self.user = name
        self._save_session(name)
        return True, f"Ingelogd als {name}."

    def logout(self) -> None:
        self.user = None
        self._save_session(None)
