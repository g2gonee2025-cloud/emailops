import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir / "backend" / "src"))
import unittest
from pathlib import Path

# Ensure backend/src is in path
sys.path.append(str(Path("backend/src").resolve()))


class TestProductionReadiness(unittest.TestCase):
    def test_s08_queue_module(self):
        print("\nTesting S08 (Queue) Availability...")
        import cortex.queue

        assert hasattr(cortex.queue, "JobQueue"), "JobQueue abstract base class missing"
        assert hasattr(
            cortex.queue, "RedisStreamsQueue"
        ), "RedisStreamsQueue impl missing"
        assert hasattr(cortex.queue, "InMemoryQueue"), "InMemoryQueue impl missing"

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
        assert Path(".pre-commit-config.yaml").exists(), "Pre-commit config missing"
        assert Path("backend/Dockerfile").exists(), "Dockerfile missing"
        print("PASS: Dockerfile and pre-commit config exist")


if __name__ == "__main__":
    unittest.main()
