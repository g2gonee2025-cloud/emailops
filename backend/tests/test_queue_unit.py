import sys
import time
from unittest.mock import MagicMock, patch

import pytest
from cortex.queue import (
    CeleryQueue,
    InMemoryQueue,
    Job,
    JobStatus,
    RedisStreamsQueue,
)


class TestJobDataClasses:
    def test_job_status_constants(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.PROCESSING == "processing"

    def test_job_serialization(self):
        job = Job(id="test-1", type="ingest", payload={"foo": "bar"}, priority=10)
        data = job.to_dict()
        assert data["id"] == "test-1"
        assert data["payload"]["foo"] == "bar"

        job2 = Job.from_dict(data)
        assert job2.id == job.id
        assert job2.payload == job.payload


class TestInMemoryQueue:
    def test_enqueue_dequeue_flow(self):
        q = InMemoryQueue()
        job_id = q.enqueue("ingest", {"a": 1}, priority=5)
        stats = q.get_queue_stats()
        assert stats["pending"] == 1

        # Dequeue
        job = q.dequeue(["ingest"])
        assert job is not None
        assert job["id"] == job_id
        assert job["type"] == "ingest"

        stats = q.get_queue_stats()
        assert stats["processing"] == 1

        # Ack
        q.ack(job_id)
        stats = q.get_queue_stats()
        assert stats["completed"] == 1

    def test_timeout(self):
        q = InMemoryQueue()
        job = q.dequeue(["ingest"], timeout=0.1)
        assert job is None

    def test_nack_retry(self):
        q = InMemoryQueue()
        job_id = q.enqueue("ingest", {})
        job = q.dequeue(["ingest"])
        q.nack(job_id, error="fail")

        # Should be requeued
        stats = q.get_queue_stats()
        assert stats["pending"] == 1

        # Dequeue again
        job = q.dequeue(["ingest"])
        assert job["attempts"] == 2

        # Fail max times
        # manually set attempts
        q._jobs[job_id].attempts = 3
        q.nack(job_id, error="fail")
        stats = q.get_queue_stats()
        assert stats["dead_letter"] == 1


class TestRedisStreamsQueue:
    def test_init_missing_redis(self):
        with patch.dict(sys.modules, {"redis": None}):
            # We need to reload or ensure import fails
            # But since import is inside __init__, it runs fresh
            with pytest.raises(ImportError):
                RedisStreamsQueue()

    def test_enqueue(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.from_url.return_value = mock_client

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            q = RedisStreamsQueue()
            q.enqueue("ingest", {"a": 1})

            mock_client.xadd.assert_called()
            mock_client.hset.assert_called()

    def test_dequeue_empty(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.from_url.return_value = mock_client
        mock_client.xreadgroup.return_value = []

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            q = RedisStreamsQueue()
            job = q.dequeue(["ingest"], timeout=1)
            assert job is None

    def test_dequeue_success(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.from_url.return_value = mock_client

        # Mock xreadgroup result
        mock_client.xreadgroup.return_value = [
            (
                "stream_key",
                [
                    (
                        "msg_id",
                        {
                            "id": "job-1",
                            "type": "ingest",
                            "payload": '{"a": 1}',
                            "priority": "0",
                        },
                    )
                ],
            )
        ]

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            q = RedisStreamsQueue()
            job = q.dequeue(["ingest"])
            assert job["id"] == "job-1"
            assert job["payload"]["a"] == 1

    def test_ack(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.from_url.return_value = mock_client
        mock_client.hgetall.return_value = {"stream": "stream", "message_id": "msg"}

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            q = RedisStreamsQueue()
            q.ack("job-1")
            mock_client.xack.assert_called_with("stream", "cortex_workers", "msg")

    def test_cleanup(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.from_url.return_value = mock_client

        # Mock scan
        mock_client.scan.side_effect = [("cursor", ["key1"]), ("0", [])]
        mock_client.hgetall.return_value = {
            "status": "completed",
            "completed_at": str(time.time() - 100000),
        }

        with patch.dict(sys.modules, {"redis": mock_redis_mod}):
            q = RedisStreamsQueue()
            cleaned = q.cleanup_completed(max_age_seconds=100)
            assert cleaned == 1
            mock_client.delete.assert_called_with("key1")


class TestCeleryQueue:
    def test_init_missing_celery(self):
        with patch.dict(sys.modules, {"celery": None}):
            with pytest.raises(ImportError):
                CeleryQueue()

    def test_init_success(self):
        mock_celery_mod = MagicMock()
        # Ensure Celery class is mockable
        mock_app = MagicMock()
        mock_celery_mod.Celery.return_value = mock_app

        with patch.dict(sys.modules, {"celery": mock_celery_mod}):
            q = CeleryQueue()
            assert q._app is not None
