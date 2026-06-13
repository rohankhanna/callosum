from __future__ import annotations

import contextlib
import hashlib
import secrets
import time
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from callosum.auth_db import ApiKey, AuthDB, Session, User, UsernameTakenError

# argon2id parameters: defaults from argon2-cffi (time_cost=2, memory=64MiB,
# parallelism=8) are appropriate for a localhost service serving humans, not
# attacker-resistant batch hashing. Keep defaults.
_HASHER = PasswordHasher()
# Precomputed decoy hash so the "username unknown" path runs the same
# verify() work as the "password wrong" path. Computed once at import.
_DECOY_HASH = _HASHER.hash("decoy-for-constant-time-username-lookup")

# Plaintext API keys are 32 bytes urandom, base64url-encoded. The prefix is
# the leading 8 chars (a hint visible in /auth/keys listings; not used for
# authentication).
_API_KEY_PREFIX_LEN = 8

# Session tokens are 32 bytes urandom, base64url-encoded. Stored only as a
# sha256 of the plaintext.
_SESSION_TOKEN_BYTES = 32
_API_KEY_BYTES = 32


@dataclass(frozen=True, slots=True)
class IssuedKey:
    """The plaintext value returned exactly once when a key is created."""

    api_key: ApiKey
    plaintext: str


@dataclass(frozen=True, slots=True)
class IssuedSession:
    session: Session
    plaintext: str


class AuthError(Exception):
    """Base class for authentication failures."""


class InvalidCredentialsError(AuthError):
    pass


class SessionInvalidError(AuthError):
    pass


class ApiKeyInvalidError(AuthError):
    pass


class AuthService:
    """High-level operations over an AuthDB.

    Storage rules:
    - Passwords: argon2id, no plaintext ever stored or logged.
    - Session tokens and API keys: 32 bytes urandom each. Plaintext is shown
      to the user once at creation; only sha256(plaintext) is in the DB.

    The service does not enforce TTLs internally beyond the values it stores;
    callers are expected to pass the same `now` semantics consistently.
    """

    def __init__(self, db: AuthDB, *, session_ttl_seconds: int = 1800) -> None:
        self._db = db
        self._session_ttl_seconds = session_ttl_seconds

    @property
    def db(self) -> AuthDB:
        return self._db

    @property
    def session_ttl_seconds(self) -> int:
        return self._session_ttl_seconds

    # ---- registration / login -----------------------------------------

    def register(self, *, username: str, password: str) -> User:
        if not username or not password:
            raise InvalidCredentialsError("username and password must be non-empty")
        password_hash = _HASHER.hash(password)
        try:
            user_id = self._db.insert_user(username=username, password_hash=password_hash, created_at=time.time())
        except UsernameTakenError:
            raise InvalidCredentialsError("username is already taken") from None
        user = self._db.get_user_by_id(user_id)
        if user is None:  # pragma: no cover - defensive
            raise RuntimeError("inserted user disappeared")
        return user

    def login(self, *, username: str, password: str) -> IssuedSession:
        user = self._db.get_user_by_username(username)
        if user is None:
            # Run a verify against the precomputed decoy hash so timing is
            # symmetric with the "password wrong" path. Without this the
            # no-such-user path returns visibly faster.
            with contextlib.suppress(VerifyMismatchError):
                _HASHER.verify(_DECOY_HASH, password)
            raise InvalidCredentialsError("invalid username or password")
        try:
            _HASHER.verify(user.password_hash, password)
        except VerifyMismatchError as exc:
            raise InvalidCredentialsError("invalid username or password") from exc
        return self._issue_session(user_id=user.id)

    def _issue_session(self, *, user_id: int) -> IssuedSession:
        plaintext = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
        token_hash = _sha256(plaintext)
        now = time.time()
        expires_at = now + self._session_ttl_seconds
        self._db.insert_session(
            token_hash=token_hash,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
        )
        session = Session(
            token_hash=token_hash,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
        )
        return IssuedSession(session=session, plaintext=plaintext)

    def resolve_session(self, plaintext: str) -> Session:
        """Look up a session by plaintext token. Raises if missing or expired."""
        token_hash = _sha256(plaintext)
        session = self._db.get_session(token_hash)
        if session is None:
            raise SessionInvalidError("session not found")
        if session.expires_at < time.time():
            self._db.delete_session(token_hash)
            raise SessionInvalidError("session expired")
        return session

    def logout(self, plaintext: str) -> None:
        self._db.delete_session(_sha256(plaintext))

    # ---- api keys ------------------------------------------------------

    def create_api_key(self, *, user_id: int, label: str | None = None) -> IssuedKey:
        plaintext = secrets.token_urlsafe(_API_KEY_BYTES)
        key_hash = _sha256(plaintext)
        prefix = plaintext[:_API_KEY_PREFIX_LEN]
        now = time.time()
        key_id = self._db.insert_api_key(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=prefix,
            label=label,
            created_at=now,
        )
        api_key = ApiKey(
            id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=prefix,
            label=label,
            created_at=now,
            last_used_at=None,
            revoked_at=None,
        )
        return IssuedKey(api_key=api_key, plaintext=plaintext)

    def list_api_keys(self, *, user_id: int) -> list[ApiKey]:
        return self._db.list_api_keys(user_id)

    def revoke_api_key(self, *, key_id: int, user_id: int) -> bool:
        return self._db.revoke_api_key(key_id=key_id, user_id=user_id, revoked_at=time.time())

    def resolve_api_key(self, plaintext: str) -> ApiKey:
        """Look up an api key by plaintext. Raises if unknown or revoked.

        Touches `last_used_at` as a best-effort side effect so users can see
        when each key was last active.
        """
        key_hash = _sha256(plaintext)
        api_key = self._db.get_api_key_by_hash(key_hash)
        if api_key is None:
            raise ApiKeyInvalidError("api key not found")
        if api_key.revoked_at is not None:
            raise ApiKeyInvalidError("api key revoked")
        self._db.touch_api_key(key_id=api_key.id, now=time.time())
        return api_key


def _sha256(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()
