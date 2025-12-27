"""
Queue abstraction for Cortex.

Implements §7.4 of the Canonical Blueprint.
Abstracts the underlying queue mechanism (Redis Streams, Celery, etc.)
as per Blueprint §2.1.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Job Status and Data Classes
# -----------------------------------------------------------------------------


class JobStatus:
    """Represents the status of a job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Job:
    """Represents a job in the queue."""

    id: str
    type: str
    payload: dict[str, Any]
    priority: int = 0
    status: str = JobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "payload": self.payload,
            "priority": self.priority,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        """Create job from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            payload=data.get("payload", {}),
            priority=data.get("priority", 0),
            status=data.get("status", JobStatus.PENDING),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------


class JobQueue(ABC):
    """Abstract base class for job queues."""

    @abstractmethod
    def enqueue(self, job_type: str, payload: dict[str, Any], priority: int = 0) -> str:
        """
        Enqueue a job.

        Args:
            job_type: Type of job (e.g., 'ingest', 'reindex')
            payload: Job payload data
            priority: Job priority (higher = more urgent)

        Returns:
            Job ID
        """
        pass

    @abstractmethod
    def dequeue(self, job_types: list[str], timeout: int = 10) -> dict[str, Any] | None:
        """
        Dequeue a job.

        Args:
            job_types: List of job types to accept
            timeout: Maximum time to wait in seconds

        Returns:
            Job dictionary or None if no job available
        """
        pass

    @abstractmethod
    def ack(self, job_id: str) -> None:
        """
        Acknowledge job completion.

        Args:
            job_id: ID of the completed job
        """
        pass

    @abstractmethod
    def nack(self, job_id: str, error: str | None = None) -> None:
        """
        Negative acknowledge (fail/retry).

        Args:
            job_id: ID of the failed job
            error: Optional error message
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """
        Get job status.

        Args:
            job_id: ID of the job

        Returns:
            Job status dictionary or None
        """
        pass

    @abstractmethod
    def get_queue_stats(self) -> dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue stats (pending, processing, etc.)
        """
        pass


# -----------------------------------------------------------------------------
# In-Memory Queue (Development/Testing)
# -----------------------------------------------------------------------------


class InMemoryQueue(JobQueue):
    """Simple in-memory queue for development/testing."""

    def __init__(self):
        from queue import PriorityQueue

        self._queue = PriorityQueue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def enqueue(self, job_type: str, payload: dict[str, Any], priority: int = 0) -> str:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            type=job_type,
            payload=payload,
            priority=priority,
        )
        with self._lock:
            self._jobs[job_id] = job
            # PriorityQueue is min-heap, so negate priority for higher = more urgent
            self._queue.put((-priority, job.created_at, job_id))
        logger.debug(f"Enqueued job {job_id} of type {job_type}")
        return job_id

    def dequeue(self, job_types: list[str], timeout: int = 10) -> dict[str, Any] | None:
        from queue import Empty

        start = time.time()
        while time.time() - start < timeout:
            try:
                with self._lock:
                    _, _, job_id = self._queue.get_nowait()
                    job = self._jobs.get(job_id)
                    if job and job.type in job_types:
                        job.status = JobStatus.PROCESSING
                        job.started_at = time.time()
                        job.attempts += 1
                        return job.to_dict()
                    elif job:
                        # Put back if not matching type
                        self._queue.put((-job.priority, job.created_at, job_id))
                        time.sleep(
                            0.1
                        )  # Prevent tight loop processing only non-matching jobs
            except Empty:
                # Avoid busy-waiting if the queue is empty
                time.sleep(0.1)
        return None

    def ack(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                logger.debug(f"Acknowledged job {job_id}")

    def nack(self, job_id: str, error: str | None = None) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.error = error
                if job.attempts >= job.max_attempts:
                    job.status = JobStatus.DEAD_LETTER
                    logger.warning(
                        f"Job {job_id} moved to dead letter after {job.attempts} attempts"
                    )
                else:
                    job.status = JobStatus.PENDING
                    self._queue.put((-job.priority, job.created_at, job_id))
                    logger.debug(f"Job {job_id} requeued (attempt {job.attempts})")

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def get_queue_stats(self) -> dict[str, int]:
        with self._lock:
            stats = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "dead_letter": 0,
            }
            for job in self._jobs.values():
                if job.status == JobStatus.PENDING:
                    stats["pending"] += 1
                elif job.status == JobStatus.PROCESSING:
                    stats["processing"] += 1
                elif job.status == JobStatus.COMPLETED:
                    stats["completed"] += 1
                elif job.status == JobStatus.FAILED:
                    stats["failed"] += 1
                elif job.status == JobStatus.DEAD_LETTER:
                    stats["dead_letter"] += 1
            return stats


# -----------------------------------------------------------------------------
# Redis Streams Queue (Production)
# -----------------------------------------------------------------------------


class RedisStreamsQueue(JobQueue):
    """
    Redis Streams-based job queue for production use.

    Implements Blueprint §7.4:
    * Reliable message delivery with consumer groups
    * Automatic dead-letter handling
    * Job status tracking
    * Priority support via multiple streams
    """

    # Stream/key naming
    STREAM_PREFIX = "cortex:jobs:"
    STATUS_PREFIX = "cortex:job_status:"
    CONSUMER_GROUP = "cortex_workers"
    DEAD_LETTER_STREAM = "cortex:dead_letter"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        job_types: list[str] | None = None,
        max_retries: int = 3,
        visibility_timeout: int = 300,  # 5 minutes
        block_timeout: int = 5000,  # 5 seconds in ms
    ):
        """
        Initialize Redis Streams queue.

        Args:
            redis_url: Redis connection URL
            max_retries: Maximum retry attempts before dead-letter
            visibility_timeout: Seconds before unacked job is reclaimed
            block_timeout: Milliseconds to block waiting for jobs
            job_types: List of all known job types (optional)
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package required for RedisStreamsQueue. "
                "Install with: pip install redis"
            )

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._max_retries = max_retries
        self._visibility_timeout = visibility_timeout
        self._block_timeout = block_timeout
        self._consumer_name = f"{socket.gethostname()}-{os.getpid()}"
        self._lock = threading.Lock()
        self._job_types = job_types or []

        # Initialize consumer groups for known job types
        if self._job_types:
            self._ensure_consumer_groups(self._job_types)

        logger.info(
            f"RedisStreamsQueue initialized with consumer: {self._consumer_name}"
        )

    def _stream_name(self, job_type: str, priority: int = 0) -> str:
        """Get stream name for job type and priority."""
        if priority > 5:
            return f"{self.STREAM_PREFIX}{job_type}:high"
        elif priority < -5:
            return f"{self.STREAM_PREFIX}{job_type}:low"
        return f"{self.STREAM_PREFIX}{job_type}:normal"

    def _ensure_consumer_groups(self, job_types: list[str]) -> None:
        """Ensure consumer groups exist for all job types and priorities."""
        priorities = ["high", "normal", "low"]
        for job_type in job_types:
            for priority in priorities:
                stream = f"{self.STREAM_PREFIX}{job_type}:{priority}"
                try:
                    self._redis.xgroup_create(
                        stream, self.CONSUMER_GROUP, id="0", mkstream=True
                    )
                    logger.debug(f"Created consumer group for stream: {stream}")
                except Exception as e:
                    # Ignore BUSYGROUP errors (group already exists)
                    if "BUSYGROUP" in str(e):
                        continue
                    logger.warning(f"Failed to create consumer group for {stream}: {e}")

    def enqueue(self, job_type: str, payload: dict[str, Any], priority: int = 0) -> str:
        """Enqueue a job to Redis Streams."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            type=job_type,
            payload=payload,
            priority=priority,
        )

        stream = self._stream_name(job_type, priority)

        # Store job data
        job_data = {
            "id": job_id,
            "type": job_type,
            "payload": json.dumps(payload),
            "priority": str(priority),
            "created_at": str(job.created_at),
            "max_attempts": str(job.max_attempts),
        }

        # Add to stream
        self._redis.xadd(stream, job_data)

        # Store job status
        self._redis.hset(
            f"{self.STATUS_PREFIX}{job_id}",
            mapping={
                "status": JobStatus.PENDING,
                "attempts": "0",
                "created_at": str(job.created_at),
                "type": job_type,
            },
        )
        self._redis.expire(f"{self.STATUS_PREFIX}{job_id}", 86400 * 7)  # 7 day TTL

        logger.debug(f"Enqueued job {job_id} to stream {stream}")
        return job_id

    def dequeue(self, job_types: list[str], timeout: int = 10) -> dict[str, Any] | None:
        """Dequeue a job from Redis Streams."""
        # Build list of streams to read from (in priority order)
        streams = {}
        for job_type in job_types:
            for priority in ["high", "normal", "low"]:
                stream = f"{self.STREAM_PREFIX}{job_type}:{priority}"
                streams[stream] = ">"  # Read new messages

        if not streams:
            return None

        try:
            # First, try to claim stale messages (pending too long)
            reclaimed = self._claim_stale_messages(job_types)
            if reclaimed:
                return reclaimed

            # Read new messages
            result = self._redis.xreadgroup(
                self.CONSUMER_GROUP,
                self._consumer_name,
                streams,
                count=1,
                block=self._block_timeout,
            )

            if not result:
                return None

            # Parse result: [(stream_name, [(message_id, {data})])]
            stream_name, messages = result[0]
            if not messages:
                return None

            message_id, data = messages[0]

            job_id = data["id"]
            job_type = data["type"]
            payload = json.loads(data["payload"])

            # Update job status
            attempts = self._redis.hincrby(
                f"{self.STATUS_PREFIX}{job_id}", "attempts", 1
            )
            self._redis.hset(
                f"{self.STATUS_PREFIX}{job_id}",
                mapping={
                    "status": JobStatus.PROCESSING,
                    "started_at": str(time.time()),
                    "message_id": message_id,
                    "stream": stream_name,
                    "consumer": self._consumer_name,
                },
            )

            return {
                "id": job_id,
                "type": job_type,
                "payload": payload,
                "priority": int(data.get("priority", 0)),
                "attempts": attempts,
                "_message_id": message_id,
                "_stream": stream_name,
            }

        except Exception as e:
            logger.error(f"Error dequeuing from Redis: {e}")
            return None

    def _claim_stale_messages(self, job_types: list[str]) -> dict[str, Any] | None:
        """Attempt to claim stale (pending too long) messages."""
        min_idle_time = self._visibility_timeout * 1000  # Convert to ms

        for job_type in job_types:
            for priority in ["high", "normal", "low"]:
                stream = f"{self.STREAM_PREFIX}{job_type}:{priority}"

                try:
                    # Get pending messages
                    pending = self._redis.xpending_range(
                        stream, self.CONSUMER_GROUP, min="-", max="+", count=10
                    )

                    for entry in pending:
                        msg_id = entry["message_id"]
                        idle_time = entry.get("time_since_delivered", 0)

                        if idle_time >= min_idle_time:
                            # Attempt to claim
                            claimed = self._redis.xclaim(
                                stream,
                                self.CONSUMER_GROUP,
                                self._consumer_name,
                                min_idle_time=min_idle_time,
                                message_ids=[msg_id],
                            )

                            if claimed:
                                msg_id, data = claimed[0]
                                job_id = data["id"]

                                # Check retry count
                                attempts = int(
                                    self._redis.hget(
                                        f"{self.STATUS_PREFIX}{job_id}", "attempts"
                                    )
                                    or 0
                                )

                                if attempts >= self._max_retries:
                                    # Move to dead letter
                                    self._move_to_dead_letter(
                                        job_id, stream, msg_id, data
                                    )
                                    continue

                                # Update status and return
                                self._redis.hincrby(
                                    f"{self.STATUS_PREFIX}{job_id}", "attempts", 1
                                )
                                self._redis.hset(
                                    f"{self.STATUS_PREFIX}{job_id}",
                                    mapping={
                                        "status": JobStatus.PROCESSING,
                                        "started_at": str(time.time()),
                                        "message_id": msg_id,
                                        "stream": stream,
                                        "consumer": self._consumer_name,
                                    },
                                )

                                return {
                                    "id": job_id,
                                    "type": data["type"],
                                    "payload": json.loads(data["payload"]),
                                    "priority": int(data.get("priority", 0)),
                                    "attempts": attempts + 1,
                                    "_message_id": msg_id,
                                    "_stream": stream,
                                }

                except Exception as e:
                    logger.warning(f"Error claiming stale messages from {stream}: {e}")

        return None

    def _move_to_dead_letter(
        self, job_id: str, stream: str, message_id: str, data: dict[str, Any]
    ) -> None:
        """Move a failed job to the dead letter queue."""
        # Add to dead letter stream
        dead_data = {
            **data,
            "original_stream": stream,
            "original_message_id": message_id,
            "dead_letter_at": str(time.time()),
        }
        self._redis.xadd(self.DEAD_LETTER_STREAM, dead_data)

        # Acknowledge original message
        self._redis.xack(stream, self.CONSUMER_GROUP, message_id)

        # Update status
        self._redis.hset(
            f"{self.STATUS_PREFIX}{job_id}",
            mapping={
                "status": JobStatus.DEAD_LETTER,
                "dead_letter_at": str(time.time()),
            },
        )

        logger.warning(f"Job {job_id} moved to dead letter queue")

    def ack(self, job_id: str) -> None:
        """Acknowledge job completion."""
        status_key = f"{self.STATUS_PREFIX}{job_id}"
        status_data = self._redis.hgetall(status_key)

        if not status_data:
            logger.warning(f"Job {job_id} not found for ack")
            return

        stream = status_data.get("stream")
        message_id = status_data.get("message_id")

        if stream and message_id:
            self._redis.xack(stream, self.CONSUMER_GROUP, message_id)

        # Update status
        self._redis.hset(
            status_key,
            mapping={
                "status": JobStatus.COMPLETED,
                "completed_at": str(time.time()),
            },
        )

        logger.debug(f"Acknowledged job {job_id}")

    def nack(self, job_id: str, error: str | None = None) -> None:
        """
        Negative acknowledge (fail/retry).

        If retries are exhausted, moves the job to the dead-letter queue.
        Otherwise, the job remains in the pending queue to be re-claimed later.
        The `_claim_stale_messages` logic handles re-delivery.
        """
        status_key = f"{self.STATUS_PREFIX}{job_id}"
        status_data = self._redis.hgetall(status_key)

        if not status_data:
            logger.warning(f"Job {job_id} not found for nack")
            return

        attempts = int(status_data.get("attempts", 0))
        stream = status_data.get("stream")
        message_id = status_data.get("message_id")

        if attempts >= self._max_retries:
            # Move to dead letter queue
            if stream and message_id:
                # Retrieve the full job data to move it
                # XCLAIM returns the message, but it might be simpler to just query it
                # We assume the message is still in the stream.
                data = self._redis.xrange(
                    stream, min=message_id, max=message_id, count=1
                )
                if data:
                    _, job_data = data[0]
                    self._move_to_dead_letter(job_id, stream, message_id, job_data)
                else:
                    # Fallback if message is gone for some reason
                    logger.warning(
                        f"Could not find message {message_id} in stream {stream} for job {job_id} to dead-letter."
                    )
                    self._redis.hset(
                        status_key,
                        mapping={
                            "status": JobStatus.DEAD_LETTER,
                            "error": error
                            or "Max retries exceeded (message not found)",
                        },
                    )
            else:
                # If we don't have stream/message info, we can't ACK, but we can update status
                self._redis.hset(
                    status_key,
                    mapping={
                        "status": JobStatus.DEAD_LETTER,
                        "error": error or "Max retries exceeded (no stream info)",
                    },
                )
        else:
            # Job will be retried. Just update its status.
            # It remains in the consumer group's pending list and will be
            # picked up again by `_claim_stale_messages` after `visibility_timeout`.
            self._redis.hset(
                status_key,
                mapping={
                    "status": JobStatus.PENDING,  # Mark as pending for retry
                    "error": error or "",
                    "last_failed_at": str(time.time()),
                },
            )
            logger.debug(f"Nacked job {job_id} (attempt {attempts}). Will be retried.")

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get job status."""
        status_data = self._redis.hgetall(f"{self.STATUS_PREFIX}{job_id}")
        if not status_data:
            return None

        return {
            "id": job_id,
            "status": status_data.get("status"),
            "attempts": int(status_data.get("attempts", 0)),
            "created_at": float(status_data.get("created_at", 0)),
            "started_at": (
                float(status_data.get("started_at", 0))
                if status_data.get("started_at")
                else None
            ),
            "completed_at": (
                float(status_data.get("completed_at", 0))
                if status_data.get("completed_at")
                else None
            ),
            "error": status_data.get("error"),
            "type": status_data.get("type"),
        }

    def get_queue_stats(self) -> dict[str, int]:
        """Get queue statistics."""
        stats = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "dead_letter": 0,
        }

        # Count dead letter messages
        try:
            dl_info = self._redis.xinfo_stream(self.DEAD_LETTER_STREAM)
            stats["dead_letter"] = dl_info.get("length", 0)
        except Exception:
            pass

        # Count messages in job streams
        job_types_to_scan = self._job_types or []
        if not job_types_to_scan:
            logger.warning(
                "Cannot get complete queue stats, job types not provided at init."
            )
        for job_type in job_types_to_scan:
            for priority in ["high", "normal", "low"]:
                stream = f"{self.STREAM_PREFIX}{job_type}:{priority}"
                try:
                    info = self._redis.xinfo_stream(stream)
                    stats["pending"] += info.get("length", 0)

                    # Get consumer group info for processing count
                    groups = self._redis.xinfo_groups(stream)
                    for group in groups:
                        if group["name"] == self.CONSUMER_GROUP:
                            stats["processing"] += group.get("pending", 0)
                except Exception:
                    pass

        return stats

    def cleanup_completed(self, max_age_seconds: int = 86400) -> int:
        """
        Clean up completed job status records.

        Args:
            max_age_seconds: Maximum age of completed jobs to keep

        Returns:
            Number of records cleaned up
        """
        # This is a maintenance operation - scan status keys and remove old completed ones
        cleaned = 0
        cursor = "0"
        cutoff = time.time() - max_age_seconds

        while True:
            cursor, keys = self._redis.scan(
                cursor=cursor, match=f"{self.STATUS_PREFIX}*", count=100
            )

            for key in keys:
                status_data = self._redis.hgetall(key)
                if status_data.get("status") == JobStatus.COMPLETED:
                    completed_at = float(status_data.get("completed_at", 0))
                    if completed_at < cutoff:
                        self._redis.delete(key)
                        cleaned += 1

            if cursor == "0":
                break

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} completed job records")

        return cleaned


# -----------------------------------------------------------------------------
# Celery-Based Queue (Alternative Production Backend)
# -----------------------------------------------------------------------------


class CeleryQueue(JobQueue):
    """
    Celery-based job queue for production use.

    Provides an alternative to Redis Streams using Celery for
    task distribution and result tracking.
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        job_types: list[str] | None = None,
    ):
        """
        Initialize Celery queue.

        Args:
            broker_url: Celery broker URL (Redis, RabbitMQ, etc.)
            job_types: List of job types to register as tasks
        """
        try:
            from celery import Celery
        except ImportError:
            raise ImportError(
                "celery package required for CeleryQueue. "
                "Install with: pip install celery[redis]"
            )

        self._app = Celery("cortex", broker=broker_url)
        self._app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            result_expires=86400,  # 1 day
            task_acks_late=True,
            task_reject_on_worker_lost=True,
        )
        self._pending_jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._job_types = job_types or []
        self._task_map: dict[str, Any] = {}

        # Register task handlers
        self._register_tasks()

        logger.info("CeleryQueue initialized")

    def _register_tasks(self) -> None:
        """Register Celery tasks for job types."""

        def create_task(name):
            @self._app.task(name=f"cortex.{name}", bind=True, max_retries=3)
            def generic_task(self, payload: dict[str, Any]) -> dict[str, Any]:
                # Worker-side logic would go here.
                # For this abstraction, we just return a success marker.
                return {"status": "completed", "payload": payload}

            return generic_task

        for job_type in self._job_types:
            self._task_map[job_type] = create_task(job_type)

    def enqueue(self, job_type: str, payload: dict[str, Any], priority: int = 0) -> str:
        """Enqueue a job via Celery."""
        job_id = str(uuid.uuid4())

        task = self._task_map.get(job_type)
        if not task:
            raise ValueError(f"Unknown or unregistered job type for Celery: {job_type}")

        # Calculate Celery priority (0-9, lower = higher priority)
        celery_priority = max(0, min(9, 5 - (priority // 2)))

        result = task.apply_async(
            args=[payload],
            task_id=job_id,
            priority=celery_priority,
        )

        with self._lock:
            self._pending_jobs[job_id] = {
                "id": job_id,
                "type": job_type,
                "payload": payload,
                "priority": priority,
                "result": result,
            }

        logger.debug(f"Enqueued Celery task {job_id} of type {job_type}")
        return job_id

    def dequeue(self, job_types: list[str], timeout: int = 10) -> dict[str, Any] | None:
        """
        Dequeue is handled by Celery workers automatically.
        This method is not used in a Celery-based setup and is here for ABC compliance.
        """
        logger.debug(
            "CeleryQueue.dequeue called but is a no-op; workers handle dequeuing."
        )
        # Block for a short time to simulate waiting, but Celery workers operate independently.
        time.sleep(timeout)
        return None

    def ack(self, job_id: str) -> None:
        """Acknowledge job completion (handled automatically by Celery)."""
        with self._lock:
            if job_id in self._pending_jobs:
                del self._pending_jobs[job_id]
        logger.debug(f"Acknowledged Celery task {job_id}")

    def nack(self, job_id: str, error: str | None = None) -> None:
        """Negative acknowledge - Celery handles retries automatically."""
        logger.debug(f"Nack Celery task {job_id}: {error}")

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get job status via Celery AsyncResult."""
        from celery.result import AsyncResult

        result = AsyncResult(job_id, app=self._app)

        status_map = {
            "PENDING": JobStatus.PENDING,
            "STARTED": JobStatus.PROCESSING,
            "SUCCESS": JobStatus.COMPLETED,
            "FAILURE": JobStatus.FAILED,
            "RETRY": JobStatus.PENDING,
        }

        return {
            "id": job_id,
            "status": status_map.get(result.status, JobStatus.PENDING),
            "result": result.result if result.ready() else None,
            "error": str(result.result) if result.failed() else None,
        }

    def get_queue_stats(self) -> dict[str, int]:
        """Get queue statistics."""
        # Celery queue inspection requires the inspect API
        try:
            inspector = self._app.control.inspect()
            active = inspector.active() or {}
            reserved = inspector.reserved() or {}
            scheduled = inspector.scheduled() or {}

            return {
                "pending": sum(len(v) for v in reserved.values())
                + sum(len(v) for v in scheduled.values()),
                "processing": sum(len(v) for v in active.values()),
                "completed": 0,  # Would need result backend query
                "failed": 0,
            }
        except Exception as e:
            logger.warning(f"Failed to get Celery stats: {e}")
            return {"pending": 0, "processing": 0, "completed": 0, "failed": 0}


# -----------------------------------------------------------------------------
# Queue Factory
# -----------------------------------------------------------------------------

_queue_instance: JobQueue | None = None
_queue_lock = threading.Lock()


def get_queue(job_types: list[str] | None = None) -> JobQueue:
    """
    Get the configured queue instance.

    Uses config to determine queue type:
    - 'memory': InMemoryQueue (default for dev)
    - 'redis': RedisStreamsQueue (production)
    - 'celery': CeleryQueue (alternative production)

    Args:
        job_types (Optional[list[str]]): For Redis, a list of all job types to ensure
                                         consumer groups are created. This is only
                                         needed by worker processes.
    """
    global _queue_instance

    with _queue_lock:
        if _queue_instance is None:
            from cortex.config.loader import get_config
            from cortex.queue_registry import get_known_job_types

            config = get_config()
            job_types = get_known_job_types()

            # Check for queue type in config (default to memory)
            queue_type = getattr(config.system, "queue_type", "memory")

            if queue_type == "redis":
                try:
                    redis_url = os.getenv(
                        "OUTLOOKCORTEX_REDIS_URL", "redis://localhost:6379"
                    )
                    # For workers, this will create consumer groups. For producers, it can be an empty list.
                    all_job_types = job_types or []
                    _queue_instance = RedisStreamsQueue(
                        redis_url=redis_url, job_types=all_job_types
                    )
                    logger.info("Initialized Redis Streams queue")
                except ImportError as e:
                    logger.warning(
                        f"Redis not available: {e}, falling back to InMemory"
                    )
                    _queue_instance = InMemoryQueue()
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize Redis queue: {e}, falling back to InMemory"
                    )
                    _queue_instance = InMemoryQueue()

            elif queue_type == "celery":
                try:
                    broker_url = os.getenv(
                        "OUTLOOKCORTEX_CELERY_BROKER", "redis://localhost:6379/0"
                    )
                    _queue_instance = CeleryQueue(
                        broker_url=broker_url, job_types=job_types
                    )
                    logger.info("Initialized Celery queue")
                except ImportError as e:
                    logger.warning(
                        f"Celery not available: {e}, falling back to InMemory"
                    )
                    _queue_instance = InMemoryQueue()
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize Celery queue: {e}, falling back to InMemory"
                    )
                    _queue_instance = InMemoryQueue()
            else:
                _queue_instance = InMemoryQueue()
                logger.info("Initialized InMemory queue (development mode)")

    return _queue_instance


def reset_queue() -> None:
    """Reset the queue instance (useful for testing)."""
    global _queue_instance
    with _queue_lock:
        _queue_instance = None
