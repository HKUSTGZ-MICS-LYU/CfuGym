"""Host header generation must roll back with the rest of Implement."""
from functools import wraps

from ...runtime import coding_session, stage_services
from . import intrinsics


def atomic_header(function):
    @wraps(function)
    def wrapped(state, *args, **kwargs):
        path = intrinsics.intrinsics_header_path(state)
        if not path:
            return function(state, *args, **kwargs)
        toolbox = stage_services._toolbox_for_stage(state, "implementation")
        if toolbox.normalize(path) != path or not toolbox.is_write_allowed(path):
            raise ValueError("intrinsics header is not an approved run target")
        session = coding_session.CodingSession(toolbox, frozenset({path}))
        try:
            update = function(state, *args, **kwargs)
            session.commit()
            return update
        except Exception:
            session.rollback()
            raise
    return wrapped
