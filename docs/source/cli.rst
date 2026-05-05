Command Line Interface
======================

``tmrl`` provides commands for users who wish to use the readily implemented example pipelines for TrackMania.

Configuration uses package YAML defaults merged with ``~/TmrlData/config/local.yaml``.
See the repo ``readme/reference_guide.md`` and ``tmrl/config/README.md``.

Examples:
---------

Launch the default training pipeline for TrackMania on 3 possibly different machines:

.. code-block:: bash

   python -m tmrl --server
   python -m tmrl --trainer
   python -m tmrl --worker

Test (deploy) the readily trained example policy for TrackMania:

.. code-block:: bash

   python -m tmrl --test

Launch the reward recorder in your own track in TrackMania:

.. code-block:: bash

   python -m tmrl --record-reward

Check that the TrackMania environment is working as expected:

.. code-block:: bash

   python -m tmrl --check-env

Benchmark the RolloutWorker in TrackMania. Set ``environment.rtgym.benchmark: true`` (e.g. in ``local.yaml`` or via ``TMRL_CONFIG_OVERRIDES``):

.. code-block:: bash

   python -m tmrl --benchmark

Print merged configuration (secrets redacted) and exit:

.. code-block:: bash

   python -m tmrl --print-config

Launch the Trainer but disable logging to wandb.ai (logging is enabled by default on the trainer):

.. code-block:: bash

   python -m tmrl --trainer --no-wandb
