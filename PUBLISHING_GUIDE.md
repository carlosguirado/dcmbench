# Publishing dcmbench to PyPI

This guide provides instructions for publishing the dcmbench package to PyPI (Python Package Index).

## Prerequisites

1. **Create PyPI Account**: If you don't have one, create an account at https://pypi.org/account/register/
2. **Create Test PyPI Account** (optional but recommended): Create an account at https://test.pypi.org/account/register/ for testing

## Setup API Token

For security, use API tokens instead of username/password:

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with scope "Entire account" or specific to dcmbench project
3. Save the token securely (you'll only see it once)

## Publishing Steps

### 1. Test on TestPyPI (Recommended)

First, test your package on TestPyPI to ensure everything works:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# When prompted:
# Username: __token__
# Password: <your-test-pypi-token>
```

Test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dcmbench
```

### 2. Publish to PyPI

Once you've verified everything works on TestPyPI:

```bash
# Upload to PyPI
python -m twine upload dist/*

# When prompted:
# Username: __token__
# Password: <your-pypi-token>
```

### 3. Configure .pypirc (Optional)

To avoid entering credentials repeatedly, create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <your-pypi-token>

[testpypi]
username = __token__
password = <your-test-pypi-token>
```

**Important**: Set file permissions to keep it secure:
```bash
chmod 600 ~/.pypirc
```

Then you can upload without entering credentials:
```bash
# For TestPyPI
python -m twine upload --repository testpypi dist/*

# For PyPI
python -m twine upload --repository pypi dist/*
```

## Version Management

Before each release:

1. Update version in `pyproject.toml` and `setup.py`
2. Update CHANGELOG if you maintain one
3. Commit changes and tag the release:
   ```bash
   git add .
   git commit -m "Release version 0.1.2"
   git tag -a v0.1.2 -m "Version 0.1.2"
   git push origin main --tags
   ```

## Building New Releases

For subsequent releases:

```bash
# Clean previous builds
rm -rf build dist *.egg-info

# Build new distribution
python -m build

# Check the distribution
python -m twine check dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Installation

Once published, users can install dcmbench with:

```bash
# Basic installation
pip install dcmbench

# With PyTorch support
pip install dcmbench[pytorch]

# For development
pip install dcmbench[dev]
```

## Package URLs

Once published, your package will be available at:
- PyPI: https://pypi.org/project/dcmbench/
- Documentation: https://dcmbench.readthedocs.io (if configured)
- GitHub: https://github.com/carlosguirado/dcmbench

## Troubleshooting

### Common Issues

1. **"Invalid distribution file"**: Run `python -m twine check dist/*` to validate files
2. **"Package already exists"**: You can't overwrite existing versions. Increment the version number
3. **Authentication failed**: Ensure you're using `__token__` as username and the correct token

### Best Practices

1. Always test on TestPyPI first
2. Use semantic versioning (MAJOR.MINOR.PATCH)
3. Include a comprehensive README.md
4. Test installation in a clean virtual environment
5. Document all dependencies clearly

## GitHub Actions (Optional)

For automated publishing, you can set up GitHub Actions:

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: python -m twine upload dist/*
```

Add your PyPI token as a GitHub secret named `PYPI_API_TOKEN`.