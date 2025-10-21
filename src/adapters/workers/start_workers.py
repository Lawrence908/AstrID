#!/usr/bin/env python3
"""Start Dramatiq workers for AstrID background processing."""

import argparse
import logging
import sys

from dramatiq.cli import main as dramatiq_main

from src.adapters.workers.config import (
    get_task_queues,
    get_worker_config,
    worker_manager,
)
from src.core.logging import configure_domain_logger

logger = configure_domain_logger("workers.startup")


def setup_workers():
    """Set up worker configuration and broker."""
    try:
        logger.info("Setting up worker configuration")

        # Get configuration
        config = get_worker_config()
        task_queues = get_task_queues()

        # Setup worker manager
        worker_manager.setup_broker(config)

        logger.info(f"Configured {len(task_queues)} task queues")
        for queue in task_queues:
            if queue.enabled:
                logger.info(f"  - {queue.queue_name} ({queue.worker_type.value})")

        return True

    except Exception as e:
        logger.error(f"Failed to setup workers: {e}")
        return False


def start_workers(
    worker_types: list[str] | None = None,
    queues: list[str] | None = None,
    processes: int = 1,
    threads: int = 4,
    log_level: str = "INFO",
):
    """Start Dramatiq workers."""
    try:
        logger.info("Starting Dramatiq workers")

        # Setup workers first
        if not setup_workers():
            return False

        # Build dramatiq command
        cmd_args = [
            "dramatiq",
            "src.adapters.workers.tasks",
            "--processes",
            str(processes),
            "--threads",
            str(threads),
            "--log-level",
            log_level,
        ]

        # Add queue filters if specified
        if queues:
            cmd_args.extend(["--queues"] + queues)

        # Add worker type filters if specified
        if worker_types:
            # Map worker types to queue names
            task_queues = get_task_queues()
            type_to_queues = {}
            for queue in task_queues:
                if queue.enabled:
                    worker_type = queue.worker_type.value
                    if worker_type not in type_to_queues:
                        type_to_queues[worker_type] = []
                    type_to_queues[worker_type].append(queue.queue_name)

            selected_queues = []
            for worker_type in worker_types:
                if worker_type in type_to_queues:
                    selected_queues.extend(type_to_queues[worker_type])
                else:
                    logger.warning(f"Unknown worker type: {worker_type}")

            if selected_queues:
                cmd_args.extend(["--queues"] + selected_queues)

        logger.info(f"Starting workers with command: {' '.join(cmd_args)}")

        # Start workers
        sys.argv = cmd_args
        dramatiq_main()

        return True

    except KeyboardInterrupt:
        logger.info("Workers stopped by user")
        return True
    except Exception as e:
        logger.error(f"Failed to start workers: {e}")
        return False


def main():
    """Main entry point for worker startup."""
    parser = argparse.ArgumentParser(description="Start AstrID Dramatiq workers")

    parser.add_argument(
        "--worker-types",
        nargs="+",
        choices=[
            "observation_ingestion",
            "preprocessing",
            "differencing",
            "detection",
            "curation",
            "notification",
        ],
        help="Worker types to start (default: all)",
    )

    parser.add_argument(
        "--queues",
        nargs="+",
        help="Specific queues to process (default: all enabled queues)",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads per process (default: 4)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup configuration, don't start workers",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("AstrID Worker Startup")
    logger.info(f"Arguments: {args}")

    if args.setup_only:
        success = setup_workers()
    else:
        success = start_workers(
            worker_types=args.worker_types,
            queues=args.queues,
            processes=args.processes,
            threads=args.threads,
            log_level=args.log_level,
        )

    if success:
        logger.info("Worker startup completed successfully")
        sys.exit(0)
    else:
        logger.error("Worker startup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
