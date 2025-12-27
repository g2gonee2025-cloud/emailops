import time
from unittest.mock import ANY, MagicMock, patch

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
        _, kwargs = mock_redis.hset.call_args
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

    def test_dequeue_respects_priority(self, mock_redis):
        q = RedisStreamsQueue()

        # Mock xreadgroup to return nothing for high prio, but a job for normal
        mock_redis.xreadgroup.return_value = [
            (
                "cortex:jobs:ingest:normal",
                [
                    (
                        "msg-id-normal",
                        {
                            "id": "job-normal",
                            "type": "ingest",
                            "payload": '{"p": "v"}',
                            "priority": "0",
                        },
                    )
                ],
            )
        ]

        job = q.dequeue(["ingest"])
        assert job is not None
        assert job["id"] == "job-normal"
        # Verify it tried to read from high-priority streams first
        streams_arg = mock_redis.xreadgroup.call_args[0][2]
        assert "cortex:jobs:ingest:high" in streams_arg

    def test_claim_stale_message_moves_to_dead_letter_on_max_retries(self, mock_redis):
        q = RedisStreamsQueue(visibility_timeout=30, max_retries=3)

        # Mock pending to find an old message only in the high-prio stream
        def xpending_side_effect(stream, group, **kwargs):
            if stream == "cortex:jobs:ingest:high":
                return [{"message_id": "msg-old", "time_since_delivered": 35000}]
            return []

        mock_redis.xpending_range.side_effect = xpending_side_effect

        # Mock claim to successfully grab it
        mock_redis.xclaim.return_value = [
            ("msg-old", {"id": "job-stale", "payload": "{}", "type": "ingest"})
        ]
        # Mock status lookup finding max attempts
        mock_redis.hget.return_value = "3"

        # Mock xreadgroup to return nothing so it doesn't interfere
        mock_redis.xreadgroup.return_value = []
        # Call dequeue, which should trigger the claim
        job = q.dequeue(["ingest"])

        assert job is None  # Should not return the dead-lettered job
        # Verify it was moved to DLQ
        mock_redis.xadd.assert_called_with(q.DEAD_LETTER_STREAM, ANY)
        # Verify original was acked on the correct stream
        mock_redis.xack.assert_called_with(
            "cortex:jobs:ingest:high", q.CONSUMER_GROUP, "msg-old"
        )

    def test_nack_moves_to_dead_letter_after_max_retries(self, mock_redis):
        q = RedisStreamsQueue(max_retries=2)

        # Simulate 3 failures
        for i in range(3):
            # Update mock for each attempt
            mock_redis.hgetall.return_value = {
                "attempts": str(i + 1),
                "status": JobStatus.PROCESSING,
                "stream": "s",
                "message_id": "m",
            }
            # Mock the xrange call needed for dead-lettering payload
            mock_redis.xrange.return_value = [("m", {"payload": "{}"})]
            q.nack("job-123")

        # After 3rd nack (since max_retries=2), it should go to dead letter
        mock_redis.xadd.assert_called_with(q.DEAD_LETTER_STREAM, ANY)
        # Verify status was updated to DEAD_LETTER
        _, kwargs = mock_redis.hset.call_args
        assert kwargs["mapping"]["status"] == JobStatus.DEAD_LETTER

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


class TestCeleryQueue:
    @pytest.fixture
    def mock_celery(self):
        """Mock the Celery library and its submodules."""
        mock_celery_module = MagicMock()
        mock_celery_class = MagicMock()
        mock_app = MagicMock()
        mock_celery_class.return_value = mock_app
        mock_celery_module.Celery = mock_celery_class

        # Mock task registration
        mock_ingest_task = MagicMock()
        mock_reindex_task = MagicMock()
        # When the mock tasks are used as decorators, they should return themselves
        mock_ingest_task.return_value = mock_ingest_task
        mock_reindex_task.return_value = mock_reindex_task
        mock_app.task.side_effect = [mock_ingest_task, mock_reindex_task]

        mock_result_module = MagicMock()
        mock_async_result = MagicMock()
        mock_result_module.AsyncResult = mock_async_result

        modules_to_patch = {
            "celery": mock_celery_module,
            "celery.result": mock_result_module,
        }

        with patch.dict("sys.modules", modules_to_patch):
            yield {
                "app": mock_app,
                "ingest_task": mock_ingest_task,
                "reindex_task": mock_reindex_task,
                "AsyncResult": mock_async_result,
            }

    def test_enqueue(self, mock_celery):
        from cortex.queue import CeleryQueue

        q = CeleryQueue()
        job_id = q.enqueue("ingest", {"foo": "bar"}, priority=5)

        assert job_id
        # Verify correct task was called
        mock_celery["ingest_task"].apply_async.assert_called_once()
        # Verify priority was translated
        _, kwargs = mock_celery["ingest_task"].apply_async.call_args
        assert "priority" in kwargs
        assert isinstance(kwargs["priority"], int)

    def test_ack(self, mock_celery):
        from cortex.queue import CeleryQueue

        q = CeleryQueue()
        job_id = q.enqueue("ingest", {"foo": "bar"})
        q.ack(job_id)

        # In-memory pending jobs should be cleared
        assert job_id not in q._pending_jobs

    def test_nack(self, mock_celery):
        from cortex.queue import CeleryQueue

        q = CeleryQueue()
        # Nack is basically a no-op for Celery as retries are automatic
        q.nack("job-123", error="test error")
        # No crash is a pass

    def test_get_job_status_completed(self, mock_celery):
        from cortex.queue import CeleryQueue

        # Mock result object
        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.result = {"output": "data"}
        mock_celery["AsyncResult"].return_value = mock_result

        q = CeleryQueue()
        status = q.get_job_status("job-123")

        assert status["status"] == JobStatus.COMPLETED
        assert status["result"] == {"output": "data"}

    def test_get_job_status_pending(self, mock_celery):
        from cortex.queue import CeleryQueue

        mock_result = MagicMock()
        mock_result.status = "PENDING"
        mock_result.ready.return_value = False
        mock_celery["AsyncResult"].return_value = mock_result

        q = CeleryQueue()
        status = q.get_job_status("job-123")

        assert status["status"] == JobStatus.PENDING
