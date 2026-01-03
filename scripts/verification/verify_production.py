import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
backend_src = root_dir / "backend" / "src"
try:
    root_real = root_dir.resolve(strict=True)
    backend_src_real = backend_src.resolve(strict=True)
    backend_src_real.relative_to(root_real)
except (FileNotFoundError, ValueError):
    backend_src_real = None
if backend_src_real and backend_src_real.is_dir():
    sys.path.append(str(backend_src_real))
import unittest


class TestProductionReadiness(unittest.TestCase):
    def test_s08_queue_module(self: "TestProductionReadiness") -> None:
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
        self.assertIsNotNone(job_id)
        job = q.dequeue(["test_job"])
        self.assertIsNotNone(job)
        self.assertEqual(job["id"], job_id)
        q.ack(job_id)

    def test_s09_observability_module(self):
        import cortex.observability

        self.assertTrue(
            hasattr(cortex.observability, "init_observability"),
            "init_observability missing",
        )
        self.assertTrue(
            hasattr(cortex.observability, "trace_operation"),
            "trace_operation decorator missing",
        )

    def test_s10_ci_configuration(self):
        self.assertTrue(
            (root_dir / ".pre-commit-config.yaml").exists(),
            "Pre-commit config missing",
        )
        self.assertTrue(
            (root_dir / "backend/Dockerfile").exists(), "Dockerfile missing"
        )


if __name__ == "__main__":
    unittest.main()
