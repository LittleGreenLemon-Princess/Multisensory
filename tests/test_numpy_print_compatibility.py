import contextlib
import io
import unittest
from simulators.interactive import SAFE_BUILTINS, NUMPY, validate_snippet

class NumpyPrintCompatibility(unittest.TestCase):
    def test_dtype_formatting_succeeds_without_relaxing_user_imports(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec("print(np.zeros(1).dtype)", {"__builtins__": dict(SAFE_BUILTINS), "np": NUMPY})
        self.assertIn("float64", output.getvalue())
        with self.assertRaises(ImportError):
            exec("import os", {"__builtins__": dict(SAFE_BUILTINS)})
        with self.assertRaises(ImportError):
            exec("import numpy._core._dtype", {"__builtins__": dict(SAFE_BUILTINS)})
        with self.assertRaises(ValueError):
            validate_snippet("print.__globals__")
