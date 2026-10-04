"""Durable, process-safe inference budget reservations."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


MICRO_USD = Decimal("0.000001")


class BudgetError(RuntimeError):
    """Base class for budget failures."""


class BudgetExceeded(BudgetError):
    """Raised before work starts when the requested reservation will not fit."""


class ReservationError(BudgetError):
    """Raised when a reservation is missing or in an invalid state."""


def to_microusd(value: Decimal | str | int | float) -> int:
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD amount must be finite and non-negative")
    return int((amount / MICRO_USD).to_integral_value(rounding=ROUND_CEILING))


def from_microusd(value: int) -> Decimal:
    return Decimal(value) * MICRO_USD


@dataclass(frozen=True)
class BudgetSnapshot:
    cap_usd: Decimal
    settled_usd: Decimal
    held_usd: Decimal
    available_usd: Decimal


class BudgetLedger:
    """SQLite ledger that reserves worst-case spend before concurrent work starts.

    Active and uncertain reservations both consume capacity. An uncertain request
    must be reconciled manually after checking provider billing; it is never
    silently released after a network timeout.
    """

    def __init__(self, path: str | Path, cap_usd: Decimal | str | int | float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_microusd = to_microusd(cap_usd)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS budget (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    cap_microusd INTEGER NOT NULL CHECK (cap_microusd >= 0)
                );
                CREATE TABLE IF NOT EXISTS reservations (
                    reservation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    reserved_microusd INTEGER NOT NULL CHECK (reserved_microusd >= 0),
                    actual_microusd INTEGER,
                    status TEXT NOT NULL CHECK (
                        status IN ('active', 'uncertain', 'settled', 'cancelled')
                    ),
                    note TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            row = connection.execute(
                "SELECT cap_microusd FROM budget WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO budget(singleton, cap_microusd) VALUES (1, ?)",
                    (self.cap_microusd,),
                )
            elif row["cap_microusd"] != self.cap_microusd:
                raise BudgetError(
                    "ledger already exists with a different cap; use the original "
                    "cap or create a new ledger"
                )

    def snapshot(self) -> BudgetSnapshot:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN status = 'settled'
                        THEN actual_microusd ELSE 0 END), 0) AS settled,
                    COALESCE(SUM(CASE WHEN status IN ('active', 'uncertain')
                        THEN reserved_microusd ELSE 0 END), 0) AS held
                FROM reservations
                """
            ).fetchone()
        settled = int(row["settled"])
        held = int(row["held"])
        available = max(0, self.cap_microusd - settled - held)
        return BudgetSnapshot(
            cap_usd=from_microusd(self.cap_microusd),
            settled_usd=from_microusd(settled),
            held_usd=from_microusd(held),
            available_usd=from_microusd(available),
        )

    def reserve(
        self,
        reservation_id: str,
        *,
        run_id: str,
        model: str,
        max_cost_usd: Decimal | str | int | float,
    ) -> None:
        requested = to_microusd(max_cost_usd)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                totals = connection.execute(
                    """
                    SELECT
                        COALESCE(SUM(CASE WHEN status = 'settled'
                            THEN actual_microusd ELSE 0 END), 0) AS settled,
                        COALESCE(SUM(CASE WHEN status IN ('active', 'uncertain')
                            THEN reserved_microusd ELSE 0 END), 0) AS held
                    FROM reservations
                    """
                ).fetchone()
                used = int(totals["settled"]) + int(totals["held"])
                if used + requested > self.cap_microusd:
                    raise BudgetExceeded(
                        f"reservation ${from_microusd(requested)} exceeds remaining "
                        f"budget ${from_microusd(max(0, self.cap_microusd - used))}"
                    )
                connection.execute(
                    """
                    INSERT INTO reservations(
                        reservation_id, run_id, model, reserved_microusd, status
                    ) VALUES (?, ?, ?, ?, 'active')
                    """,
                    (reservation_id, run_id, model, requested),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def reconcile(
        self,
        reservation_id: str,
        actual_cost_usd: Decimal | str | int | float,
        *,
        note: str | None = None,
    ) -> None:
        actual = to_microusd(actual_cost_usd)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM reservations WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    raise ReservationError(f"unknown reservation {reservation_id!r}")
                if row["status"] not in ("active", "uncertain"):
                    raise ReservationError(
                        f"reservation {reservation_id!r} is already {row['status']}"
                    )
                exceeded = actual > int(row["reserved_microusd"])
                connection.execute(
                    """
                    UPDATE reservations
                    SET actual_microusd = ?, status = 'settled', note = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE reservation_id = ?
                    """,
                    (actual, note, reservation_id),
                )
                connection.commit()
                if exceeded:
                    raise ReservationError(
                        f"actual cost ${from_microusd(actual)} exceeded reserved "
                        f"amount ${from_microusd(row['reserved_microusd'])}; actual "
                        "cost was recorded and further runs may be blocked"
                    )
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise

    def mark_uncertain(self, reservation_id: str, note: str) -> None:
        self._transition(reservation_id, "uncertain", note)

    def cancel(self, reservation_id: str, note: str | None = None) -> None:
        self._transition(reservation_id, "cancelled", note)

    def _transition(self, reservation_id: str, status: str, note: str | None) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT status FROM reservations WHERE reservation_id = ?",
                    (reservation_id,),
                ).fetchone()
                if row is None:
                    raise ReservationError(f"unknown reservation {reservation_id!r}")
                if row["status"] != "active":
                    raise ReservationError(
                        f"reservation {reservation_id!r} is already {row['status']}"
                    )
                connection.execute(
                    """
                    UPDATE reservations SET status = ?, note = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE reservation_id = ?
                    """,
                    (status, note, reservation_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
