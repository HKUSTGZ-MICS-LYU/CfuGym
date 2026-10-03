"""Catch unresolved globals left behind when a helper moves between modules."""
import builtins
from pathlib import Path
import symtable
import unittest


class RuntimeBindingsTest(unittest.TestCase):
    def test_agent_modules_have_no_unbound_referenced_globals(self):
        root = Path(__file__).resolve().parents[1] / "cfu_design_agent"
        module_globals = {"__file__", "__name__", "__package__", "__spec__", "__loader__",
                          "__builtins__", "__path__", "__doc__", "__annotations__"}
        for path in sorted(root.rglob("*.py")):
            table = symtable.symtable(path.read_text(), str(path), "exec")
            defined = set(table.get_identifiers()) | set(dir(builtins)) | module_globals
            pending = [table]
            unresolved = []
            while pending:
                scope = pending.pop()
                unresolved.extend(f"{symbol.get_name()} in {scope.get_name()}:{scope.get_lineno()}"
                                  for symbol in scope.get_symbols()
                                  if symbol.is_referenced() and symbol.is_global()
                                  and symbol.get_name() not in defined)
                pending.extend(scope.get_children())
            with self.subTest(module=path.relative_to(root).as_posix()):
                self.assertEqual(unresolved, [], "helper must be reached through its owning module")


if __name__ == "__main__":
    unittest.main()
