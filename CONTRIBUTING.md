# Contributing to FeatureCraft

Thank you for your interest in contributing to FeatureCraft! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Security Issues](#security-issues)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to maintainers@featurecraft.dev.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- pip and virtualenv

### Areas for Contribution

We welcome contributions in the following areas:

- **Bug fixes:** Fix issues reported in GitHub Issues
- **New features:** Implement feature requests or propose new ones
- **Documentation:** Improve docs, add examples, fix typos
- **Tests:** Increase test coverage, add edge case tests
- **Performance:** Optimize slow operations
- **Code quality:** Refactor complex code, improve type hints

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/FeatureCraft.git
cd FeatureCraft

# Add upstream remote
git remote add upstream https://github.com/assaab/FeatureCraft.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install in editable mode with all dependencies
pip install -e ".[dev,ai,extras,schema]"

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Run linters
ruff check src tests
black --check src tests
mypy src/featurecraft

# Run security checks
bandit -r src/featurecraft
pip-audit
```

## Making Changes

### Branch Naming

Create a descriptive branch name:

```bash
git checkout -b feature/add-xyz-encoder
git checkout -b fix/issue-123-memory-leak
git checkout -b docs/improve-quickstart
git checkout -b refactor/simplify-pipeline-builder
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add frequency-based encoder for high-cardinality features
fix: resolve memory leak in out-of-fold target encoding
docs: add example for custom transformers
test: add integration tests for AI features
refactor: extract pipeline building logic into separate methods
perf: optimize categorical encoding with vectorization
chore: update dependencies to latest versions
```

### Code Changes

1. **Keep changes focused:** One feature/fix per PR
2. **Add tests:** All new code should have tests
3. **Update docs:** Update relevant documentation
4. **Type hints:** Add type annotations for new functions
5. **Docstrings:** Use Google-style docstrings

Example function:

```python
def encode_categories(
    X: pd.DataFrame,
    columns: list[str],
    strategy: str = "onehot"
) -> pd.DataFrame:
    """Encode categorical columns using specified strategy.
    
    Args:
        X: Input DataFrame with categorical columns
        columns: List of column names to encode
        strategy: Encoding strategy ("onehot", "target", "frequency")
        
    Returns:
        DataFrame with encoded columns
        
    Raises:
        ValueError: If strategy is not supported
        
    Example:
        >>> df = pd.DataFrame({"cat": ["A", "B", "A"]})
        >>> encode_categories(df, ["cat"], strategy="onehot")
        ...
    """
    # Implementation
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=featurecraft --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py

# Run specific test
pytest tests/test_pipeline.py::test_fit_basic

# Run tests matching pattern
pytest -k "encoder"

# Run only fast tests (skip slow)
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

### Writing Tests

```python
import pytest
import pandas as pd
from featurecraft import AutoFeatureEngineer

def test_pipeline_fit_basic():
    """Test that pipeline fits on simple dataset."""
    # Arrange
    X = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    y = pd.Series([0, 1, 0])
    engineer = AutoFeatureEngineer()
    
    # Act
    engineer.fit(X, y, estimator_family="tree")
    
    # Assert
    assert engineer.pipeline_ is not None
    assert engineer.feature_names_ is not None
    assert len(engineer.feature_names_) > 0

@pytest.mark.slow
def test_pipeline_large_dataset():
    """Test pipeline on large dataset (marked as slow)."""
    # ... test with large data
```

### Test Markers

- `@pytest.mark.slow` - Slow tests (> 1 second)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.ai` - Tests requiring AI API keys

## Code Style

### Formatting

We use **Black** for code formatting (line length: 100):

```bash
# Format code
black src tests examples

# Check formatting
black --check src tests
```

### Linting

We use **Ruff** for linting:

```bash
# Run linter
ruff check src tests

# Auto-fix issues
ruff check --fix src tests
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Type check
mypy src/featurecraft
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Run manually on all files
pre-commit run --all-files

# Skip hooks (not recommended)
git commit --no-verify
```

## Submitting Changes

### Pull Request Process

1. **Sync with upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature
   ```

3. **Create Pull Request:**
   - Go to GitHub and create PR from your fork
   - Fill out the PR template
   - Link related issues (`Fixes #123`)

4. **Code Review:**
   - Address reviewer feedback
   - Keep PR updated with main branch
   - Be responsive to comments

### PR Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Code is formatted (`black`, `ruff`)
- [ ] Type hints added where appropriate
- [ ] Docstrings added/updated
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] PR description explains changes clearly
- [ ] Linked to related issue(s)

### Review Process

- PRs require at least one approval from a maintainer
- CI must pass (tests, linting, security checks)
- Large changes may require design discussion first
- We aim to review PRs within 3-5 business days

## Reporting Bugs

### Before Submitting

1. **Search existing issues:** Check if already reported
2. **Try latest version:** Ensure bug exists in latest release
3. **Minimal reproduction:** Create smallest example that reproduces bug

### Bug Report Template

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- **Description:** What happened vs. what you expected
- **Environment:** OS, Python version, FeatureCraft version
- **Reproduction steps:** Minimal code to reproduce
- **Error messages:** Full stack traces
- **Additional context:** Screenshots, data samples (sanitized)

## Feature Requests

We welcome feature requests! Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- **Problem:** What problem does this solve?
- **Proposed solution:** How should it work?
- **Alternatives:** Other solutions you considered
- **Examples:** Code examples of how it would be used

## Security Issues

**DO NOT open public issues for security vulnerabilities.**

Report security issues privately:
- Email: maintainers@featurecraft.dev
- GitHub Security Advisory: [Create advisory](https://github.com/assaab/FeatureCraft/security/advisories/new)

See [SECURITY.md](SECURITY.md) for details.

## Development Guidelines

### Project Structure

```
FeatureCraft/
├── src/featurecraft/       # Main package
│   ├── pipeline.py          # Core orchestration
│   ├── config.py            # Configuration
│   ├── encoders.py          # Categorical encoders
│   ├── transformers.py      # Custom transformers
│   └── ...
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
└── templates/               # HTML templates
```

### Adding New Features

1. **Discuss first:** Open an issue to discuss design
2. **Write tests:** TDD is encouraged
3. **Update docs:** Add to relevant docs files
4. **Add example:** Include usage example
5. **Update CHANGELOG:** Document user-facing changes

### Code Complexity

Aim for:
- Functions: < 50 lines
- Cyclomatic complexity: < 10
- Cognitive complexity: < 15

For complex logic, break into smaller functions with clear names.

### Performance

- Profile before optimizing
- Add benchmarks for critical paths
- Document performance characteristics

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release: `git tag v1.0.0`
5. Push tag: `git push origin v1.0.0`
6. GitHub Actions handles PyPI publish

## Getting Help

- **Questions:** Open a [Discussion](https://github.com/assaab/FeatureCraft/discussions)
- **Bugs:** Open an [Issue](https://github.com/assaab/FeatureCraft/issues)
- **Chat:** Join our community (link TBD)
- **Email:** maintainers@featurecraft.dev

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` (all contributors)
- GitHub releases (per-release contributors)
- CHANGELOG.md (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to FeatureCraft!** 🚀

