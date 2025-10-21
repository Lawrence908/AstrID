"""Mock Dramatiq broker for testing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any


class MockMessage:
    """Mock Dramatiq message."""

    def __init__(self, message_id: str, actor_name: str, args: tuple, kwargs: dict):
        self.message_id = message_id
        self.actor_name = actor_name
        self.args = args
        self.kwargs = kwargs
        self.options = {}
        self.queue_name = "default"
        self.priority = 0
        self.delay = 0
        self.retries = 0
        self.max_retries = 20
        self.created_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.failed_at: float | None = None
        self.status = "enqueued"

    def __repr__(self) -> str:
        return f"MockMessage(id={self.message_id}, actor={self.actor_name})"


class MockActor:
    """Mock Dramatiq actor."""

    def __init__(self, fn: Callable, actor_name: str, queue_name: str = "default"):
        self.fn = fn
        self.actor_name = actor_name
        self.queue_name = queue_name
        self.options = {}

    def send(self, *args: Any, **kwargs: Any) -> MockMessage:
        """Send a message to the actor."""
        message_id = f"msg_{int(time.time() * 1000000)}"
        message = MockMessage(message_id, self.actor_name, args, kwargs)
        message.queue_name = self.queue_name
        return message

    def send_with_options(
        self,
        args: tuple | None = None,
        kwargs: dict | None = None,
        delay: int | None = None,
        **options: Any,
    ) -> MockMessage:
        """Send a message with specific options."""
        message = self.send(*(args or ()), **(kwargs or {}))
        if delay:
            message.delay = delay
        message.options.update(options)
        return message

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Direct call to the actor function."""
        return self.fn(*args, **kwargs)


class MockDramatiqBroker:
    """Mock Dramatiq broker for testing."""

    def __init__(self):
        self.actors: dict[str, MockActor] = {}
        self.queues: dict[str, list[MockMessage]] = {"default": []}
        self.messages: dict[str, MockMessage] = {}
        self.dead_letter_queue: list[MockMessage] = []
        self.processing_messages: dict[str, MockMessage] = {}
        self.completed_messages: dict[str, MockMessage] = {}
        self.failed_messages: dict[str, MockMessage] = {}
        self.middleware_stack: list[Any] = []
        self.error_on_operation: dict[str, bool] = {}
        self.auto_process = False

    def add_middleware(self, middleware: Any) -> None:
        """Add middleware to the broker."""
        self.middleware_stack.append(middleware)

    def actor(
        self,
        fn: Callable | None = None,
        *,
        actor_name: str | None = None,
        queue_name: str = "default",
        priority: int = 0,
        **options: Any,
    ) -> Callable:
        """Decorator to register an actor."""

        def decorator(func: Callable) -> MockActor:
            name = actor_name or func.__name__
            actor = MockActor(func, name, queue_name)
            actor.options.update(options)
            actor.options["priority"] = priority

            self.actors[name] = actor

            # Ensure queue exists
            if queue_name not in self.queues:
                self.queues[queue_name] = []

            return actor

        if fn is None:
            return decorator
        return decorator(fn)

    def enqueue(self, message: MockMessage) -> MockMessage:
        """Enqueue a message."""
        if self.error_on_operation.get("enqueue", False):
            raise Exception("Simulated enqueue error")

        self.messages[message.message_id] = message

        # Add to appropriate queue
        queue_name = message.queue_name
        if queue_name not in self.queues:
            self.queues[queue_name] = []

        # Insert with priority consideration
        self.queues[queue_name].append(message)
        self.queues[queue_name].sort(key=lambda m: m.priority, reverse=True)

        if self.auto_process:
            asyncio.create_task(self._process_message(message))

        return message

    async def _process_message(self, message: MockMessage) -> None:
        """Process a message (for auto-processing mode)."""
        try:
            message.status = "processing"
            message.started_at = time.time()
            self.processing_messages[message.message_id] = message

            # Simulate processing delay
            await asyncio.sleep(0.1)

            # Get the actor and execute
            actor = self.actors.get(message.actor_name)
            if actor:
                result = actor.fn(*message.args, **message.kwargs)
                if asyncio.iscoroutine(result):
                    await result

            message.status = "completed"
            message.completed_at = time.time()
            self.completed_messages[message.message_id] = message

            # Remove from processing
            self.processing_messages.pop(message.message_id, None)

        except Exception:
            message.status = "failed"
            message.failed_at = time.time()
            self.failed_messages[message.message_id] = message
            self.processing_messages.pop(message.message_id, None)

            # Add to dead letter queue if max retries exceeded
            if message.retries >= message.max_retries:
                self.dead_letter_queue.append(message)

    def get_queue_size(self, queue_name: str = "default") -> int:
        """Get the size of a queue."""
        return len(self.queues.get(queue_name, []))

    def peek_queue(
        self, queue_name: str = "default", count: int = 1
    ) -> list[MockMessage]:
        """Peek at messages in a queue without removing them."""
        queue = self.queues.get(queue_name, [])
        return queue[:count]

    def dequeue(self, queue_name: str = "default") -> MockMessage | None:
        """Dequeue the next message from a queue."""
        queue = self.queues.get(queue_name, [])
        if queue:
            message = queue.pop(0)
            return message
        return None

    def flush_all(self) -> None:
        """Flush all queues."""
        for queue in self.queues.values():
            queue.clear()

    def flush_queue(self, queue_name: str) -> None:
        """Flush a specific queue."""
        if queue_name in self.queues:
            self.queues[queue_name].clear()

    def get_message(self, message_id: str) -> MockMessage | None:
        """Get a message by ID."""
        return self.messages.get(message_id)

    def get_message_status(self, message_id: str) -> str | None:
        """Get the status of a message."""
        message = self.get_message(message_id)
        return message.status if message else None

    def retry_message(self, message_id: str) -> bool:
        """Retry a failed message."""
        message = self.failed_messages.get(message_id)
        if message and message.retries < message.max_retries:
            message.retries += 1
            message.status = "enqueued"
            message.failed_at = None

            # Re-enqueue the message
            self.queues[message.queue_name].append(message)
            self.failed_messages.pop(message_id, None)
            return True
        return False

    def cancel_message(self, message_id: str) -> bool:
        """Cancel a message."""
        message = self.get_message(message_id)
        if message and message.status in ["enqueued", "processing"]:
            message.status = "cancelled"

            # Remove from queue if enqueued
            for queue in self.queues.values():
                if message in queue:
                    queue.remove(message)
                    break

            # Remove from processing if processing
            self.processing_messages.pop(message_id, None)
            return True
        return False

    def get_actor(self, actor_name: str) -> MockActor | None:
        """Get an actor by name."""
        return self.actors.get(actor_name)

    def list_actors(self) -> list[str]:
        """List all registered actors."""
        return list(self.actors.keys())

    def list_queues(self) -> list[str]:
        """List all available queues."""
        return list(self.queues.keys())

    def enable_auto_processing(self) -> None:
        """Enable automatic message processing."""
        self.auto_process = True

    def disable_auto_processing(self) -> None:
        """Disable automatic message processing."""
        self.auto_process = False

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation for specific operations."""
        self.error_on_operation[operation] = should_error

    def clear_all_data(self) -> None:
        """Clear all broker data."""
        self.actors.clear()
        for queue in self.queues.values():
            queue.clear()
        self.messages.clear()
        self.dead_letter_queue.clear()
        self.processing_messages.clear()
        self.completed_messages.clear()
        self.failed_messages.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get broker statistics."""
        total_enqueued = sum(len(queue) for queue in self.queues.values())

        return {
            "actors": len(self.actors),
            "queues": len(self.queues),
            "total_enqueued": total_enqueued,
            "processing": len(self.processing_messages),
            "completed": len(self.completed_messages),
            "failed": len(self.failed_messages),
            "dead_letter": len(self.dead_letter_queue),
            "queue_sizes": {name: len(queue) for name, queue in self.queues.items()},
        }
