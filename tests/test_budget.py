from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autoresearch_budget.budget import (
    BudgetExceeded,
    BudgetLedger,
    ReservationError,
)


class BudgetLedgerTest(unittest.TestCase):
    def test_corrects_settled_cost_from_provider_billing(self):
        with TemporaryDirectory() as temp:
            ledger = BudgetLedger(Path(temp) / "budget.sqlite3", "1.00")
            ledger.reserve("r1", run_id="run", model="model", max_cost_usd="0.50")
            ledger.reconcile("r1", "0")
            ledger.correct_settled("r1", "0.125", note="provider billing")
            self.assertEqual(ledger.snapshot().settled_usd, Decimal("0.125000"))

    def test_reservations_hold_capacity_and_reconcile_actual_cost(self):
        with TemporaryDirectory() as temp:
            ledger = BudgetLedger(Path(temp) / "budget.sqlite3", "1.00")
            ledger.reserve(
                "run-1", run_id="run-1", model="mock/small", max_cost_usd="0.70"
            )
            self.assertEqual(ledger.snapshot().available_usd, Decimal("0.300000"))
            with self.assertRaises(BudgetExceeded):
                ledger.reserve(
                    "run-2",
                    run_id="run-2",
                    model="mock/frontier",
                    max_cost_usd="0.31",
                )
            ledger.reconcile("run-1", "0.20")
            snapshot = ledger.snapshot()
            self.assertEqual(snapshot.settled_usd, Decimal("0.200000"))
            self.assertEqual(snapshot.held_usd, Decimal("0.000000"))
            self.assertEqual(snapshot.available_usd, Decimal("0.800000"))

    def test_uncertain_reservation_stays_held(self):
        with TemporaryDirectory() as temp:
            ledger = BudgetLedger(Path(temp) / "budget.sqlite3", "1")
            ledger.reserve(
                "run-1", run_id="run-1", model="mock/small", max_cost_usd="0.4"
            )
            ledger.mark_uncertain("run-1", "connection closed after dispatch")
            self.assertEqual(ledger.snapshot().held_usd, Decimal("0.400000"))
            ledger.reconcile("run-1", "0.1")
            self.assertEqual(ledger.snapshot().settled_usd, Decimal("0.100000"))

    def test_actual_cost_may_not_exceed_reservation(self):
        with TemporaryDirectory() as temp:
            ledger = BudgetLedger(Path(temp) / "budget.sqlite3", "1")
            ledger.reserve(
                "run-1", run_id="run-1", model="mock/small", max_cost_usd="0.1"
            )
            with self.assertRaises(ReservationError):
                ledger.reconcile("run-1", "0.100001")
            self.assertEqual(ledger.snapshot().settled_usd, Decimal("0.100001"))


if __name__ == "__main__":
    unittest.main()
