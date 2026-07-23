Installation
============

.. warning::

    This package is **not published on PyPI**. Running ``pip install pykeen`` installs upstream
    PyKEEN, which does **not** contain the hierarchy extensions documented here. Always install
    from the source archive as described below. The import name stays ``pykeen``, so existing
    PyKEEN code runs unchanged.

Python 3.9 or newer is required.

Linux and Mac Users
-------------------

Download this repository as a ZIP archive and extract it. Create a virtual environment, then
install from the extracted directory (or pass its path in place of ``.``).

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    $ uv venv
    $ source .venv/bin/activate
    $ uv pip install -e .

Or with pip:

.. code-block:: bash

    $ python3 -m venv .venv
    $ source .venv/bin/activate
    $ pip install -e .

``uv sync`` is an alternative to the uv commands above: it creates ``.venv`` and installs the
exact versions pinned in the checked-in ``uv.lock`` rather than resolving fresh.

Extras use bracket notation, e.g. ``pip install -e ".[plotting]"``; see :ref:`extras` below.

Verify the installation with:

.. code-block:: bash

    $ pykeen version

Windows Users
-------------

Windows support is experimental, and installing PyTorch is less straightforward. Install
`Anaconda <https://www.anaconda.com/>`_ and follow the instructions on the `PyTorch website
<https://pytorch.org/get-started/locally/>`_ first, then proceed as above with the Windows
activation script:

.. code-block:: bat

    > uv venv
    > .venv\Scripts\activate
    > uv pip install -e .

If you're having trouble with ``pip`` or ``sqlite``, you might also have to run ``conda install
pip setuptools wheel sqlite``.

Development
-----------

The regular installation above already uses editable mode (``-e``), so local changes to the
source take effect immediately. To include the development dependencies, install the
corresponding extras, e.g. ``pip install -e ".[tests,docs]"``.

If you're interested in making contributions, please see ``CONTRIBUTING.md`` in the repository root.

To automatically ensure compliance to our style guide, please install pre-commit hooks using the following code block
from in the same directory.

.. code-block:: bash

    $ pip install pre-commit
    $ pre-commit install

.. _extras:

Extras
------

PyKEEN has several extras for installation that are defined in the ``[project.optional-dependencies]`` section of
``pyproject.toml``. They can be included with installation using the bracket notation like in
``pip install -e ".[docs]"``. Several can be listed, comma-delimited like in
``pip install -e ".[docs,plotting]"``. The same syntax works with ``uv pip install``.

================ =========================================================================================
Name             Description
================ =========================================================================================
``templating``   Building of templated documentation, like the README
``plotting``     Plotting with ``seaborn`` and generation of word clouds
``mlflow``       Tracking of results with ``mlflow``
``wandb``        Tracking of results with ``wandb``
``neptune``      Tracking of results with ``neptune``
``tensorboard``  Tracking of results with :mod:`tensorboard` via :mod:`torch.utils.tensorboard`
``transformers`` Label-based initialization with ``transformers``.
``tests``        Code needed to run tests. Typically handled with ``tox -e py``
``docs``         Building of the documentation
``opt_einsum``   Improve performance of :func:`torch.einsum` by replacing with :func:`opt_einsum.contract`
``biomedicine``  Use of :mod:`pyobo` for lookup of biomedical entity labels
================ =========================================================================================
