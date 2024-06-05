from experiments.dj.evaluation_tables import FlowPriorEval

FlowPriorEval.populate(
    display_progress=True, suppress_errors=True, reserve_jobs=True, order="random"
)
