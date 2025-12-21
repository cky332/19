CI/CD Testing
=============

MarkDiffusion includes a continuous integration testing system via GitHub Actions (workflow: ``selective-tests.yml``). 
Due to the extensive functionality of the repository, running the complete test suite takes considerable time. 
Therefore, our CI/CD pipeline focuses on essential tests to ensure code quality while maintaining efficiency.

The CI/CD tests use a lightweight test suite located in ``tests_ci/`` directory, which contains the same test 
structure as the full test suite but runs faster by focusing on initialization and interface validation.

Workflow Overview
-----------------

The CI/CD workflow automatically detects changed files and runs targeted tests based on the modifications:

Core Framework Changes
~~~~~~~~~~~~~~~~~~~~~~

When changes are detected in core framework files:

- ``watermark/auto_watermark.py``
- ``watermark/base.py``
- ``watermark/auto_config.py``
- ``watermark/__init__.py``

The system runs initialization tests for **all algorithms** (``--skip-generation --skip-detection``), ensuring 
the framework logic remains correct and compatible across all watermarking methods without running the 
time-consuming generation and detection processes.

Algorithm-Specific Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When changes are detected in:

- Specific algorithm folders (e.g., ``watermark/tr/``, ``watermark/gs/``)
- Algorithm configuration files in ``config/`` (e.g., ``TR.json``, ``GS.json``)

The system automatically identifies the affected algorithms and runs initialization tests for **only those 
specific algorithms**, ensuring efficient testing while maintaining quality assurance.

Evaluation and Test Module Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When changes are detected in:

- ``evaluation/`` module
- ``tests_ci/`` directories

The system runs fast evaluation tests (``tests_ci/test_pipelines.py``) to verify that the evaluation 
pipelines and tools function correctly.

Workflow Triggers
-----------------

The CI/CD workflow is triggered on:

- **Pull Requests**: Automatically runs on all pull requests
- **Push to Main**: Runs when code is pushed to the main branch
- **Manual Dispatch**: Can be manually triggered, which forces full test scope (all algorithms + evaluation)

Benefits
--------

This targeted testing approach provides several advantages:

- **Fast Feedback**: Developers receive quick feedback on their changes
- **Resource Efficiency**: Only necessary tests are run, saving computational resources
- **Comprehensive Coverage**: Critical changes trigger appropriate test coverage
- **Quality Assurance**: Ensures that framework modifications don't break existing functionality
- **Intelligent Detection**: Automatically identifies which algorithms are affected by changes


For the complete test suite with full coverage (including generation and detection tests), 
see :doc:`comprehensive_test`.
