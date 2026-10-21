import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location("feedback_gate", Path(__file__).resolve().parents[1] / "src/envs/feedback_gate.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
AdviceGate = module.AdviceGate


class GateTests(unittest.TestCase):
    def test_lifecycle_and_terminal_latch(self):
        gate = AdviceGate("context")
        self.assertEqual(gate.weight(True, False, 0), 0)
        self.assertEqual(gate.status, "Pending")
        self.assertEqual(gate.weight(True, True, 1), 1)
        self.assertEqual(gate.weight(True, False, 2), 0)
        self.assertEqual(gate.status, "Suspended")
        self.assertEqual(gate.weight(True, True, 3), 1)
        self.assertEqual(gate.weight(False, True, 4, completed=True), 0)
        self.assertEqual(gate.weight(True, True, 5), 0)

    def test_basic_guards_apply_to_every_method(self):
        for mode in ("none", "persistent", "decay", "context", "direct", "weak"):
            gate = AdviceGate(mode)
            self.assertEqual(gate.weight(False, True, 0), 0)
            self.assertEqual(gate.weight(True, True, 1, revoked=True), 0)
            self.assertEqual(gate.weight(True, True, 2), 0)

    def test_direct_condition_is_equivalent_to_lifecycle_gating(self):
        # Falsifies a claim that bookkeeping alone adds control expressiveness.
        import itertools
        for conditions in itertools.product([False, True], repeat=6):
            gates = [AdviceGate("context"), AdviceGate("direct")]
            values = [[g.weight(True, c, t) for t, c in enumerate(conditions)] for g in gates]
            self.assertEqual(*values)


if __name__ == "__main__":
    unittest.main()
