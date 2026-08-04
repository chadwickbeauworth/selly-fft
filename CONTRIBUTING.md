# Contributing to selly-fft

Thank you for your interest in contributing to selly-fft! This project is
guided by the principle:

> **dE/dt = β(C − D)E** — minimize division (D), maximize cooperation (C),
> and let benevolence (E) grow exponentially.

## How to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/selly-fft.git
   cd selly-fft
   ```
3. **Install** in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your change:
   ```bash
   git checkout -b feature/your-feature
   ```
5. **Write code** following the conventions below.
6. **Run tests** to ensure everything passes:
   ```bash
   python3 -m pytest tests/ -v
   ```
7. **Commit** and **push** to your fork, then open a **Pull Request**.

## Development Guidelines

### Code Style
- Type hints throughout (Python 3.10+).
- NumPy-style docstrings with mathematical formulas in reStructuredText.
- Keep dependencies minimal: only `numpy` at runtime.
- Use the `src/` layout; tests live in `tests/`.

### Patent Safety
- **Do not implement** multidimensional extensions (covered by active
  US11561951B2).
- **Do not implement** bidirectional or phase-rotated modulation
  (covered by active US10438690B2).
- **Do not market** this as a "quantum computer" — the term is
  metaphorical in the patent literature.
- If you are unsure whether a feature falls under active claims,
  **don't implement it** and open an issue instead.

### Algorithm Corrections
- This library deliberately deviates from the Run-112 design spec's
  normalization and correlation method (see `Run-126` and `PATENTS.md`).
- Any improvement to the algorithm should be:
  1. Documented with mathematical justification.
  2. Accompanied by tests proving the improvement.
  3. Added to `PRIOR-ART.md` as a disclosure.

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with benchmark output
python3 -m pytest tests/test_benchmark.py -v -s

# Run a specific test file
python3 -m pytest tests/test_core.py -v
```

## Reporting Issues

When reporting an issue, please include:
- The version of selly-fft you are using.
- Your operating system and Python version.
- A minimal reproduction of the issue.
- Any relevant output or error messages.

## dE/dt Principle

All contributions are evaluated against the Love Equation: **dE/dt = β(C − D)E**.
Does this contribution increase cooperation and decrease division? If it
improves the library while staying within expired patent claims and
maintaining mathematical correctness, it aligns with the project's intent.
