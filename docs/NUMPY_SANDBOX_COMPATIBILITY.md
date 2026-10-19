# NumPy array-method compatibility

Robot snippets may perform ordinary numerical work using NumPy, including
`array.sum()`, `array.mean()` and axis reductions. NumPy implements several
`ndarray` methods using a lazy C-level import of its private `_methods` module.
That import can inherit the snippet's restricted builtins and was incorrectly
rejected as an unauthorized user import.

The compatibility bridge resolves this one numerical implementation module in
the host and returns it only for the C-level empty-list `fromlist` signature.
Ordinary Python import statements use `None` or tuples and remain rejected for
private modules. The bridge accepts neither arbitrary NumPy submodules nor
relative imports. NumPy 1.x and 2.x use different names for this same module;
the host resolves the available spelling. The AST validator still forbids
private imports, dynamic `__import__`, private attribute access, file writes,
and reflection in snippets. This is a targeted compatibility repair, not a
claim that in-process Python is a general security boundary for hostile code.

Regression tests run reductions in fresh Python interpreters. Running an
equivalent reduction in the host first can populate NumPy's C caches and hide
the original failure. Coverage includes masks, slices, empty arrays, axes,
`keepdims`, `dtype`, `out`, helper functions, and rejection of nonpublic imports
and file/reflection capabilities. The older dtype-print regression is retained.

The isolated patch changes no simulator, action primitive, score, budget, or
agent knowledge. Applying it to future experiments requires recording the new
source version. Do not hot-patch an already running frozen campaign or merge
pre-repair and post-repair measurements without identifying the conditions.
