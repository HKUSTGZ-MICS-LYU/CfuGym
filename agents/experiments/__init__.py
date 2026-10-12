"""Experiment harness for the CFU design agent: shared infrastructure + run presets.

``harness/`` holds the shared experiment infrastructure (matrix expansion,
sbt lock shim, execution, collection, failure classification, aggregation);
the run scripts are thin presets over it: ``run_matrix.py`` (pass@1/3/5
matrix), ``run_ablation.py`` (capability ladder) and ``run_sweep.py`` (every
embench-iot benchmark once, in order).  Every preset forwards the arguments
after a bare ``--`` verbatim to ``agents/run_cfu_agent.py``.
"""
