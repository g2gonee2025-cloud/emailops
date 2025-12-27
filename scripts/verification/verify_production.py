import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
backend_src = root_dir / "backend" / "src"
try:
    root_real = root_dir.resolve(strict=True)
    backend_src_real = backend_src.resolve(strict=True)
    backend_src_real.relative_to(root_real)
except Exception:
    backend_src_real = None
if backend_src_real and backend_src_real.is_dir():
    sys.path.append(str(backend_src_real))
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

# Ensure backend/src is in path
sys.path.append(str(Path("backend/src").resolve()))


class TestProductionReadiness(unittest.TestCase):
    def test_s08_queue_module(self: "TestProductionReadiness") -> None:
        print("\nTesting S08 (Queue) Availability...")
        import cortex.queue

        self.assertTrue(
            hasattr(cortex.queue, "JobQueue"), "JobQueue abstract base class missing"
        )
        self.assertTrue(
            hasattr(cortex.queue, "RedisStreamsQueue"), "RedisStreamsQueue impl missing"
        )
        self.assertTrue(
            hasattr(cortex.queue, "InMemoryQueue"), "InMemoryQueue impl missing"
        )

        # Test basic in-memory queue logic
        q = cortex.queue.InMemoryQueue()
        job_id = q.enqueue("test_job", {"foo": "bar"}, priority=10)
        assert job_id is not None
        job = q.dequeue(["test_job"])
        assert job is not None
        assert job["id"] == job_id
        q.ack(job_id)
        print("PASS: Queue module validates")

    def test_s09_observability_module(self):
        print("\nTesting S09 (Observability) Availability...")
        import cortex.observability

        assert hasattr(
            cortex.observability, "init_observability"
        ), "init_observability missing"
        assert hasattr(
            cortex.observability, "trace_operation"
        ), "trace_operation decorator missing"
        print("PASS: Observability module validates")

    def test_s10_ci_configuration(self):
        print("\nTesting S10 (CI/CD) Configuration...")
        self.assertTrue(
            Path(".pre-commit-config.yaml").exists(), "Pre-commit config missing"
        )
        self.assertTrue(Path("backend/Dockerfile").exists(), "Dockerfile missing")
        print("PASS: Dockerfile and pre-commit config exist")


if __name__ == "__main__":
    unittest.main()
