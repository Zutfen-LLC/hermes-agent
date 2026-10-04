"""The tool-only build stage imports PM without third-party dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_minimal_bootstrap_closure_reaches_pm_paths_and_locks(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    stage = tmp_path / "stage"
    shutil.copytree(repo / "pm", stage / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "hermes_constants.py", stage / "hermes_constants.py")
    (stage / "hermes_cli").mkdir()
    for name in ("__init__.py", "runtime_state.py"):
        shutil.copy2(repo / "hermes_cli" / name, stage / "hermes_cli" / name)
    store = stage / "tools"
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"),
               HERMES_RUNTIME_DIR=str(store), PYTHONPATH=str(stage))
    script = """
from pathlib import Path
import pm.paths
from pm.store import Store
from pm.lock import Facts
root = pm.paths.store_root()
with Store(root).install_lock():
    facts = Facts(root / 'facts.json')
    facts.record_state('probe', 'checked', [])
assert facts.path.is_file()
print(root)
"""
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=stage, env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == store


def test_managed_python_signing_import_needs_no_pm_or_application(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = """
import sys
class NoApplication:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pm' or fullname == 'utils' or fullname.startswith(('pm.', 'agent.')):
            raise AssertionError('signing imported ' + fullname)
sys.meta_path.insert(0, NoApplication())
from hermes_cli.macos_signing import sign_managed_python
assert callable(sign_managed_python)
"""
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=repo,
                            env=dict(os.environ, HERMES_HOME=str(tmp_path / "home")),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.platforms("linux", "macos", "windows")
def test_runtime_staging_streams_uv_output_without_tomllib(tmp_path):
    """Bootstrap runs before PM selects its Python: Docker has 3.10 and
    historical Windows updaters have 3.11, without os.set_blocking for pipes.
    """
    repo = Path(__file__).resolve().parents[2]
    script = """
import os
import sys
if sys.platform == 'win32' and hasattr(os, 'set_blocking'):
    del os.set_blocking
class NoTomllib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('tomllib', 'pm.workspace', 'pm.plugin_declarations'):
            raise AssertionError('runtime staging imported ' + fullname)
sys.meta_path.insert(0, NoTomllib())
import pm.runtime_stage
from pm.environment import _run_streaming
import subprocess
result = _run_streaming([sys.executable, '-c', 'print(\"no solution found\")'],
                        cwd='.', env={}, timeout=30, output=sys.stderr)
assert result.returncode == 0, result
"""
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=repo,
                            env=dict(os.environ, HERMES_HOME=str(tmp_path / "home")),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_update_maintenance_tail_imports_without_application_graph(tmp_path):
    """Cold-bootstrap completion runs the maintenance tail on PM's interpreter with a
    minimal app recipe and a reduced source snapshot: only the i18n kernel of ``agent/``
    is staged (installer e2e contract), and importing ``hermes_cli.main`` would drag in
    the full application graph (dotenv & co.). The tail must therefore resolve its
    project root through the stdlib fast path and import neither ``hermes_cli.main``
    nor any ``agent`` module beyond the i18n kernel. Failures stay loud: an import that
    escapes this closure still raises ModuleNotFoundError here.
    """
    repo = Path(__file__).resolve().parents[2]
    allowed_agent = {"agent", "agent.jiter_preload", "agent.i18n",
                     "agent.i18n_layers", "agent.i18n_languages"}
    script = """
import sys
ALLOWED = %ALLOWED%
class NoApplicationGraph:
    \"\"\"Model the cold shape: the application graph and its third-party deps
    (dotenv) are ABSENT, so their imports raise ModuleNotFoundError exactly as a
    cold completion child sees them — an import that escapes the closure fails
    loudly here, it is not swallowed.\"\"\"
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'hermes_cli.main' or fullname.startswith('hermes_cli.main.'):
            raise ModuleNotFoundError('maintenance tail imported ' + fullname, name=fullname)
        if fullname.startswith('dotenv'):
            # The dependency is genuinely absent in the cold shape: report "not found"
            # the way a bare interpreter does (find_spec probing is how the fallback
            # decides; probing is not importing).
            return None
        if fullname == 'agent' or fullname.startswith('agent.'):
            if fullname not in ALLOWED:
                raise ModuleNotFoundError('maintenance tail imported ' + fullname, name=fullname)
        return None
sys.meta_path.insert(0, NoApplicationGraph())
import hermes_cli.update_cmd_maint as tail
# The tail resolves its root through the cold fallback (main/dotenv unavailable).
root = tail._project_root()
assert root.name
# Localization during bootstrap still resolves: the staged i18n kernel formats an en key
# without any further agent imports. Under ``-S`` ruamel is absent, so the catalog layer
# degrades to the bare key by design (i18n's documented last-resort fallback, logged, not
# raised); what must NOT happen is an import escaping the closure. Assert the degraded
# resolution is the bare key — deterministic — rather than a translated string that would
# depend on the outer environment.
import agent.i18n
translated = agent.i18n.t('cli.shared.n_more', lang='en', count=5)
assert translated == 'cli.shared.n_more', translated
# The version lookups survive the cold shape: no pyproject at the resolved root is a
# legitimate None, and a missing file must not raise out of the tail.
version = tail._read_project_version()
assert version is None or isinstance(version, str)
print('tail-closure-ok')
""".replace("%ALLOWED%", repr(sorted(allowed_agent)))
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=repo,
                            env=dict(os.environ, HERMES_HOME=str(tmp_path / "home")),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tail-closure-ok" in result.stdout
