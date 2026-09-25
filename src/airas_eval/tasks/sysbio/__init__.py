"""Systems biology: inferring the structure of a biochemical model from
simulated experiments.

* ``reaction_network_inference`` — reactions removed from a model, recovered
  from perturbation time series (the SciGym protocol).
"""

from airas_eval.tasks.sysbio import reaction_network_inference

TASKS = (reaction_network_inference.TASK,)
