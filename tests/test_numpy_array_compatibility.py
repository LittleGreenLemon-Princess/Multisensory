"""Cold-interpreter regressions for the actual robot-snippet sandbox."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

from simulators.interactive import NUMPY, SAFE_BUILTINS, numerical_import, validate_snippet


class NumpyArrayCompatibility(unittest.TestCase):
    def run_cold(self, snippet):
        # Each case must start cold: host-side NumPy reductions can populate
        # NumPy's C caches and conceal the restricted-import regression.
        root = Path(__file__).resolve().parents[1]
        program = textwrap.dedent("""
            from pathlib import Path
            import simulators.interactive as sandbox
            assert Path(sandbox.__file__).resolve().parent.parent == Path.cwd()
            code = SNIPPET
            sandbox.validate_snippet(code)
            namespace = {"__builtins__": dict(sandbox.SAFE_BUILTINS), "np": sandbox.NUMPY}
            exec(compile(code, "<agent-program>", "exec"), namespace)
        """).replace("SNIPPET", repr(textwrap.dedent(snippet)))
        env = dict(os.environ, PYTHONPATH=str(root))
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=root, env=env,
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cold_array_reductions(self):
        cases = {
            "sum": "a.sum() == 10",
            "mean": "a.mean() == 2.5",
            "min": "a.min() == 1",
            "max": "a.max() == 4",
            "var": "a.var() == 1.25",
            "std": "abs(a.std() - np.sqrt(1.25)) < 1e-12",
            "prod": "a.prod() == 24",
            "any": "a.any()",
            "all": "a.all()",
        }
        for name, check in cases.items():
            with self.subTest(method=name):
                self.run_cold("a = np.array([1., 2., 3., 4.])\nassert " + check)

    def test_axis_keepdims_dtype_and_out(self):
        self.run_cold("""
            import numpy as np
            a = np.array([[1, 2], [3, 4]], dtype=np.int32)
            result = a.sum(axis=0, dtype=np.float64, keepdims=True)
            assert result.shape == (1, 2)
            assert np.allclose(result, [[4., 6.]])
            out = np.zeros(2)
            returned = a.sum(axis=1, out=out)
            assert returned is out
            assert np.allclose(out, [3., 7.])
            assert np.allclose(a.mean(axis=1), [1.5, 3.5])
        """)

    def test_masks_slices_and_empty_arrays(self):
        self.run_cold("""
            pixels = np.arange(12).reshape(3, 4)
            mask = pixels > 5
            assert mask.sum() == 6
            assert np.allclose(mask[:, ::2].sum(axis=0), [1, 2])
            assert np.array([], dtype=np.float64).sum() == 0
            assert not np.array([], dtype=np.bool_).any()
            assert np.array([], dtype=np.bool_).all()
        """)

    def test_function_api_and_persistent_namespace(self):
        self.run_cold("""
            import math
            a = np.array([1., 2.])
            assert np.sum(a) == a.sum() == 3
            assert np.mean(a) == a.mean() == 1.5
            def calculate(values):
                return values.sum(), values.mean()
            assert calculate(a) == (3, 1.5)
            a = a + 2
            assert calculate(a) == (7, 3.5)
            assert math.sqrt(4) == 2
        """)

    def test_explicit_nonpublic_imports_remain_denied(self):
        for statement in (
            "import os", "import sys", "import pathlib", "import importlib",
            "import numpy._core._methods", "import numpy.core._methods",
            "from numpy._core._methods import _sum", "from . import numpy",
        ):
            with self.subTest(statement=statement):
                with self.assertRaises(ValueError):
                    validate_snippet(statement)
                # Also retain the import guard when validation is accidentally
                # bypassed; ordinary Python IMPORT_NAME must not be allowed.
                with self.assertRaises(ImportError):
                    exec(statement, {"__builtins__": dict(SAFE_BUILTINS)})

    def test_file_and_reflection_capabilities_remain_unavailable(self):
        for code in (
            "open('file')", "np.array([1]).tofile('file')",
            "np.array([1]).dump('file')", "print.__globals__",
            "np.__dict__", "__import__('os')", "getattr(np, 'load')",
        ):
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_snippet(code)
        self.assertNotIn("open", SAFE_BUILTINS)
        self.assertFalse(hasattr(NUMPY, "load"))
        self.assertFalse(hasattr(NUMPY, "save"))

    def test_internal_bridge_does_not_allow_other_modules_or_relative_imports(self):
        for name, fromlist, level in (
            ("os", [], 0), ("numpy._core._dtype", [], 0),
            ("numpy._core._methods", [], 1),
            ("numpy._core._methods", ("_sum",), 0),
            ("numpy._core._methods", ["_sum"], 0),
            ("numpy._core._methods", None, 0),
        ):
            with self.subTest(name=name, fromlist=fromlist, level=level):
                with self.assertRaises(ImportError):
                    numerical_import(name, {}, {}, fromlist, level)


if __name__ == "__main__":
    unittest.main()
