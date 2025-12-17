import importlib.util
import sys
from pathlib import Path


def load_verify_idempotency_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "verify_idempotency.py"

    spec = importlib.util.spec_from_file_location(
        "verify_idempotency_module", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, repo_root


def test_verify_idempotency_runs_with_local_repo(tmp_path):
    module, repo_root = load_verify_idempotency_module()

    backend_src = (repo_root / "backend" / "src").resolve()
    assert str(backend_src) in sys.path

    conv_root = tmp_path / "test_conversations"
    module.setup_test_env(conv_root)

    report = module.scan_and_refresh(conv_root)

    assert report.manifests_created == 0
    assert report.manifests_updated == 0
