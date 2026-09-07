"""Run the unchanged independent review assertions against the imported app.

Fixtures are byte-identical review artifacts, including historical standalone
negative-control hash guards. Importing them runs no guards or tests; unittest
collects their original TestCase classes here. Baked-image identity is enforced
separately by the release runner, never by overlaying application source.
"""
import importlib.util
import logging
from pathlib import Path
import sys

_FIXTURES = Path(__file__).parent / "fixtures" / "revision2"

def _load(name):
    spec = importlib.util.spec_from_file_location(name, _FIXTURES / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # Standalone probes silence logging at import. Preserve the surrounding
    # suite's logging assertions rather than weakening or skipping them.
    root = logging.getLogger()
    handlers, level, disabled = root.handlers[:], root.level, root.manager.disable
    try:
        spec.loader.exec_module(module)
    finally:
        root.handlers = handlers
        root.setLevel(level)
        logging.disable(disabled)
    return module

# These two unmodified review modules import one another by historical name.
# Only fill absent entries, so the separate original all-provider suite can
# retain its own identical helper when run in the same interpreter.
for _name in ("protocol_socket_all", "test_protocol_all"):
    if _name not in sys.modules:
        sys.modules[_name] = _load(_name)

Review = _load("test_real_lock").Review
Independent = _load("test_adversarial").Independent
Wire = _load("test_wire_edges").Wire
