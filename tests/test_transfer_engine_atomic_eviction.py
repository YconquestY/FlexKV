"""
Unit tests for atomic indexer eviction in TransferEngine.

These tests verify that:
1. TransferOp.pending_count defaults to 1.
2. _finalize_op is called only when pending_count reaches 0.
3. With indexer enabled: CompletedOp is NOT emitted until both main KV and indexer
   workers complete (pending_count == 0).
4. With indexer disabled: behavior is identical to the original (pending_count starts
   at 1, _finalize_op is called immediately after main KV completes).
"""
import queue
import unittest
from typing import List
from unittest.mock import MagicMock, patch, call

import numpy as np

from flexkv.common.transfer import TransferOp, TransferType, CompletedOp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_op(transfer_type: TransferType = TransferType.D2H) -> TransferOp:
    """Create a minimal TransferOp for testing."""
    return TransferOp(
        graph_id=0,
        transfer_type=transfer_type,
        src_block_ids=np.array([0, 1], dtype=np.int64),
        dst_block_ids=np.array([2, 3], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Tests – TransferOp.pending_count field
# ---------------------------------------------------------------------------

class TestTransferOpPendingCount(unittest.TestCase):
    """Requirement 5: TransferOp supports pending_count field."""

    def test_default_pending_count_is_one(self):
        """pending_count SHALL default to 1 (req 5.1)."""
        op = _make_op()
        self.assertEqual(op.pending_count, 1)

    def test_pending_count_is_mutable(self):
        """pending_count SHALL be mutable (dataclass, not frozen)."""
        op = _make_op()
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)
        op.pending_count -= 1
        self.assertEqual(op.pending_count, 1)
        op.pending_count -= 1
        self.assertEqual(op.pending_count, 0)


# ---------------------------------------------------------------------------
# Tests – _finalize_op logic (unit-level, no real workers)
# ---------------------------------------------------------------------------

class TestFinalizeOpLogic(unittest.TestCase):
    """
    Requirement 1, 3, 4: _finalize_op is called only when pending_count == 0.
    We test the logic directly by simulating what _scheduler_loop does.
    """

    def _simulate_worker_done(self, op: TransferOp, finished_ops: List[TransferOp],
                               finalize_fn) -> None:
        """Simulate what _scheduler_loop does when a worker completes an op."""
        op.pending_count -= 1
        if op.pending_count == 0:
            finalize_fn(op, finished_ops)

    def test_no_indexer_finalize_called_immediately(self):
        """Without indexer: pending_count starts at 1, finalize called after main KV done (req 6.1)."""
        op = _make_op()
        self.assertEqual(op.pending_count, 1)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)

        # pending_count should be 0 and finalize should have been called once
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)

    def test_with_indexer_finalize_not_called_after_main_kv_only(self):
        """With indexer: finalize NOT called when only main KV completes (req 3.1, 4.1)."""
        op = _make_op()
        # Simulate _assign_op_to_worker incrementing pending_count before submitting to indexer
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes first
        self._simulate_worker_done(op, finished_ops, finalize_mock)

        # pending_count should be 1, finalize should NOT have been called
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()
        self.assertEqual(len(finished_ops), 0)

    def test_with_indexer_finalize_called_after_both_complete(self):
        """With indexer: finalize called exactly once when both workers complete (req 3.2, 4.2)."""
        op = _make_op()
        # Simulate _assign_op_to_worker incrementing pending_count before submitting to indexer
        op.pending_count += 1
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Main KV worker completes first
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()

        # Indexer worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)

    def test_with_indexer_finalize_called_once_regardless_of_order(self):
        """Finalize called exactly once even if indexer completes before main KV (req 3.2, 4.2)."""
        op = _make_op()
        op.pending_count += 1  # indexer registered
        self.assertEqual(op.pending_count, 2)

        finalize_mock = MagicMock()
        finished_ops: List[TransferOp] = []

        # Indexer worker completes first
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 1)
        finalize_mock.assert_not_called()

        # Main KV worker completes
        self._simulate_worker_done(op, finished_ops, finalize_mock)
        self.assertEqual(op.pending_count, 0)
        finalize_mock.assert_called_once_with(op, finished_ops)


# ---------------------------------------------------------------------------
# Tests – _finalize_op method behavior
# ---------------------------------------------------------------------------

class TestFinalizeOpMethod(unittest.TestCase):
    """
    Test that _finalize_op correctly calls free_op_from_buffer, puts CompletedOp,
    appends to finished_ops, and deletes from op_id_to_op.
    """

    def _make_engine_stub(self):
        """Create a minimal stub of TransferEngine with the real _finalize_op method."""
        from flexkv.transfer.transfer_engine import TransferEngine, free_op_from_buffer

        engine = object.__new__(TransferEngine)
        engine.op_id_to_op = {}
        engine.completed_queue = MagicMock()
        engine.pin_buffer = MagicMock()
        engine.cache_config = MagicMock()
        engine.cache_config.tokens_per_block = 16
        engine.model_config = MagicMock()
        engine.model_config.token_size_in_bytes = 2
        return engine

    def test_finalize_op_releases_buffer_and_notifies(self):
        """_finalize_op SHALL call free_op_from_buffer and put CompletedOp (req 3.2, 4.2)."""
        from flexkv.transfer.transfer_engine import TransferEngine, free_op_from_buffer

        engine = self._make_engine_stub()
        op = _make_op()
        engine.op_id_to_op[op.op_id] = op

        finished_ops: List[TransferOp] = []

        with patch('flexkv.transfer.transfer_engine.free_op_from_buffer') as mock_free:
            engine._finalize_op(op, finished_ops)

        # free_op_from_buffer called once
        mock_free.assert_called_once_with(op, engine.pin_buffer)
        # CompletedOp put to completed_queue once
        engine.completed_queue.put.assert_called_once()
        completed_op_arg = engine.completed_queue.put.call_args[0][0]
        self.assertIsInstance(completed_op_arg, CompletedOp)
        self.assertEqual(completed_op_arg.graph_id, op.graph_id)
        self.assertEqual(completed_op_arg.op_id, op.op_id)
        # op appended to finished_ops
        self.assertIn(op, finished_ops)
        # op removed from op_id_to_op
        self.assertNotIn(op.op_id, engine.op_id_to_op)

    def test_finalize_op_removes_op_from_tracking_dict(self):
        """_finalize_op SHALL delete op from op_id_to_op (req 3.2 - no double free)."""
        engine = self._make_engine_stub()
        op = _make_op()
        engine.op_id_to_op[op.op_id] = op

        finished_ops: List[TransferOp] = []

        with patch('flexkv.transfer.transfer_engine.free_op_from_buffer'):
            engine._finalize_op(op, finished_ops)

        self.assertNotIn(op.op_id, engine.op_id_to_op)

    def test_finalize_op_not_called_twice(self):
        """op_id_to_op deletion prevents double finalization (req 3.2 - exactly once)."""
        engine = self._make_engine_stub()
        op = _make_op()
        engine.op_id_to_op[op.op_id] = op

        finished_ops: List[TransferOp] = []

        with patch('flexkv.transfer.transfer_engine.free_op_from_buffer'):
            engine._finalize_op(op, finished_ops)
            # Second call should raise KeyError since op was already removed
            with self.assertRaises(KeyError):
                engine._finalize_op(op, finished_ops)


if __name__ == "__main__":
    unittest.main()
