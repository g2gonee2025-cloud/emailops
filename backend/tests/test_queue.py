import time
from unittest.mock import MagicMock, patch

import pytest
from cortex.queue import InMemoryQueue, JobStatus, RedisStreamsQueue


class TestInMemoryQueue:
    def test_enqueue_dequeue(self):
        q = InMemoryQueue()
        job_id = q.enqueue("ingest", {"foo": "bar"}, priority=1)

        assert job_id
        stats = q.get_queue_stats()
        assert stats["pending"] == 1

        # Dequeue
        job = q.dequeue(["ingest"])
        assert job is not None
        assert job["id"] == job_id
        assert job["payload"] == {"foo": "bar"}
        assert job["status"] == JobStatus.PROCESSING

    def test_ack(self):
        q = InMemoryQueue()
        job_id = q.enqueue("test", {})
        q.dequeue(["test"])
        q.ack(job_id)

        status = q.get_job_status(job_id)
        assert status["status"] == JobStatus.COMPLETED

    def test_nack_retry(self):
        q = InMemoryQueue()
        job_id = q.enqueue("test", {})
        q.dequeue(["test"])

        # Nack -> retry
        q.nack(job_id, error="failed")

        status = q.get_job_status(job_id)
        assert status["status"] == JobStatus.PENDING
        assert status["attempts"] == 1
        assert status["error"] == "failed"

        # Check it can be dequeued again
        job = q.dequeue(["test"])
        assert job is not None
        assert job["id"] == job_id
        assert job["attempts"] == 2

    def test_nack_dead_letter(self):
        q = InMemoryQueue()
        job_id = q.enqueue("test", {})

        # Manually set attempts to max
        # (Private implementation detail access for test speed)
        with q._lock:
            q._jobs[job_id].attempts = 3

        q.nack(job_id, error="fatal")

        status = q.get_job_status(job_id)
        assert status["status"] == JobStatus.DEAD_LETTER


class TestRedisStreamsQueue:
    @pytest.fixture
    def mock_redis(self):
        with patch("redis.from_url") as mock_from_url:
            mock_client = MagicMock()
            mock_from_url.return_value = mock_client
            yield mock_client

    def test_init_creates_groups(self, mock_redis):
        RedisStreamsQueue(
            redis_url="redis://localhost", job_types=["ingest", "reindex"]
        )
        # Check consumer groups created for ingest/reindex types
        assert mock_redis.xgroup_create.call_count >= 6  # 2 types * 3 priorities

    def test_enqueue(self, mock_redis):
        q = RedisStreamsQueue(redis_url="redis://localhost")
        job_id = q.enqueue("ingest", {"data": 1}, priority=5)

        assert job_id
        mock_redis.xadd.assert_called_once()
        mock_redis.hset.assert_called_once()  # Status set

    def test_dequeue_empty(self, mock_redis):
        q = RedisStreamsQueue(job_types=["ingest"])
        mock_redis.xreadgroup.return_value = []

        job = q.dequeue(["ingest"])
        assert job is None

    def test_dequeue_success(self, mock_redis):
        q = RedisStreamsQueue(job_types=["ingest"])

        # Mock xreadgroup response
        mock_redis.xreadgroup.return_value = [
            (
                "cortex:jobs:ingest:normal",
                [
                    (
                        "msg-id-1",
                        {
                            "id": "job-123",
                            "type": "ingest",
                            "payload": '{"foo": "bar"}',
                            "priority": "0",
                        },
                    )
                ],
            )
        ]

        job = q.dequeue(["ingest"])
        assert job["id"] == "job-123"
        assert job["payload"] == {"foo": "bar"}
        # ensure status updated
        mock_redis.hset.assert_called()

    def test_ack(self, mock_redis):
        q = RedisStreamsQueue()

        # Mock status lookup finding the job
        mock_redis.hgetall.return_value = {
            "stream": "test-stream",
            "message_id": "msg-123",
            "status": JobStatus.PROCESSING,
        }

        q.ack("job-123")

        mock_redis.xack.assert_called_with("test-stream", q.CONSUMER_GROUP, "msg-123")
        mock_redis.hset.assert_called()

    def test_nack_retry(self, mock_redis):
        q = RedisStreamsQueue()
        mock_redis.hgetall.return_value = {
            "attempts": "1",
            "status": JobStatus.PROCESSING,
        }

        q.nack("job-123", error="oops")

        # Verify status reset to PENDING (no dead letter yet)
        args, kwargs = mock_redis.hset.call_args
        assert kwargs["mapping"]["status"] == JobStatus.PENDING

    def test_nack_dead_letter(self, mock_redis):
        q = RedisStreamsQueue(max_retries=3)
        mock_redis.hgetall.return_value = {
            "attempts": "3",
            "status": JobStatus.PROCESSING,
            "stream": "s",
            "message_id": "m",
        }
        mock_redis.xrange.return_value = [("m", {"data": "foo"})]

        q.nack("job-123")

        # Verify call to Dead Letter Stream
        mock_redis.xadd.assert_called()
        assert mock_redis.xadd.call_args[0][0] == q.DEAD_LETTER_STREAM

    def test_claim_stale_messages(self, mock_redis):
        q = RedisStreamsQueue(visibility_timeout=30, job_types=["ingest"])

        # Mock pending messages (older than timeout)
        mock_redis.xpending_range.return_value = [
            {"message_id": "msg-old", "time_since_delivered": 35000}
        ]

        # Mock successful claim
        mock_redis.xclaim.return_value = [
            (
                "msg-old",
                {
                    "id": "job-123",
                    "type": "ingest",
                    "payload": '{"p": "v"}',
                    "priority": "0",
                },
            )
        ]

        # Mock job status lookup for retries
        mock_redis.hget.return_value = "0"

        job = q.dequeue(["ingest"])

        assert job is not None
        assert job["id"] == "job-123"
        # Check that xclaim was called
        mock_redis.xclaim.assert_called()

    def test_queue_stats(self, mock_redis):
        q = RedisStreamsQueue(job_types=["ingest", "reindex"])
        mock_redis.xinfo_stream.return_value = {"length": 5}
        mock_redis.xinfo_groups.return_value = [
            {"name": q.CONSUMER_GROUP, "pending": 2}
        ]

        stats = q.get_queue_stats()
        # 2 types * 3 priorities * 5 length = 30 pending??
        # Actually logic sums lengths.
        # We mock xinfo_stream returns length=5 for each stream call.
        # There are 6 streams (ingest/reindex * high/normal/low).
        # So 6 * 5 = 30 pending.
        assert stats["pending"] == 30
        assert stats["processing"] == 12  # 6 * 2

    def test_cleanup_completed(self, mock_redis):
        q = RedisStreamsQueue()
        # Mock scan to return one batch of keys
        mock_redis.scan.side_effect = [("0", ["status:1", "status:2"])]

        # status:1 is old, status:2 is new
        t_now = time.time()
        mock_redis.hgetall.side_effect = [
            {"status": JobStatus.COMPLETED, "completed_at": str(t_now - 100000)},
            {"status": JobStatus.COMPLETED, "completed_at": str(t_now - 10)},
        ]

        cleaned = q.cleanup_completed(max_age_seconds=50000)

        assert cleaned == 1
        mock_redis.delete.assert_called_with("status:1")
