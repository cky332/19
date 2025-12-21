# Building MarkDiffusion Documentation

This guide explains how to build the documentation locally.

## Prerequisites

1. Python 3.10 or higher
2. Sphinx and related packages

## Installation

Install documentation dependencies:

```bash
cd docs
pip install -r requirements.txt
```

## Building HTML Documentation

### On Linux/Mac:

```bash
cd docs
make html
```

### On Windows:

```bash
cd docs
make.bat html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

## Building Other Formats

### PDF

```bash
make latexpdf
```

### ePub

```bash
make epub
```

### All formats

```bash
make html epub latexpdf
```

## Cleaning Build Files

```bash
make clean
```

## Live Reload (Development)

For development with auto-reload:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Then open http://127.0.0.1:8000 in your browser.

## Read the Docs

The documentation is automatically built and hosted on Read the Docs when you push to the main branch.

Configuration file: `.readthedocs.yaml`

## Troubleshooting

### Missing Dependencies

If you get import errors:

```bash
pip install -r requirements.txt
pip install -r docs/requirements.txt
```

### Build Warnings

To see detailed warnings:

```bash
make html SPHINXOPTS="-W"
```

To fail on warnings:

```bash
make html SPHINXOPTS="-W --keep-going"
```

### Clear Cache

Sometimes you need to clear the build cache:

```bash
make clean
rm -rf docs/_build
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation page
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── tutorial.rst           # Tutorials
├── user_guide/           # User guides
│   ├── algorithms.rst
│   ├── watermarking.rst
│   ├── visualization.rst
│   └── evaluation.rst
├── advanced/             # Advanced topics
│   ├── custom_algorithms.rst
│   ├── evaluation_pipelines.rst
│   └── configuration.rst
├── api/                  # API reference
│   ├── watermark.rst
│   ├── visualization.rst
│   ├── utils.rst
│   └── evaluation.rst
├── contributing.rst      # Contributing guide
├── citation.rst          # Citation information
├── _static/             # Static files (CSS, images)
├── _templates/          # Custom templates
├── Makefile             # Build script (Unix)
├── make.bat             # Build script (Windows)
└── requirements.txt     # Documentation dependencies
```

## Contributing to Documentation

See [Contributing Guide](contributing.rst) for details on:

- Writing documentation
- Adding new pages
- Updating API docs
- Style guidelines

## Links

- Documentation: https://markdiffusion.readthedocs.io/
- GitHub: https://github.com/THU-BPM/MarkDiffusion
- Read the Docs Project: https://readthedocs.org/projects/markdiffusion/

