from experiments.dj.evaluation_tables import SBVGPEval, SIEval

SIEval.populate(
    display_progress=True, suppress_errors=True, reserve_jobs=True, order="random"
)
SBVGPEval.populate(
    display_progress=True, suppress_errors=True, reserve_jobs=True, order="random"
)
