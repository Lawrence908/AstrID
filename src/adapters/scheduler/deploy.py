"""Deployment script for Prefect flows."""

from typing import Any, cast

from prefect import serve

from src.adapters.scheduler.config import get_deployment_config, get_prefect_config
from src.adapters.scheduler.flows.model_training import (
    scheduled_retraining_flow,
    train_model_flow,
)
from src.adapters.scheduler.flows.monitoring import (
    daily_report_flow,
    system_monitoring_flow,
)
from src.adapters.scheduler.flows.process_new import run as process_new_flow
from src.core.logging import configure_domain_logger

logger = configure_domain_logger("scheduler.deploy")


def create_observation_processing_deployment():
    """Create deployment for observation processing flow."""
    config = get_prefect_config()
    deploy_config = get_deployment_config()

    return process_new_flow.to_deployment(
        name="observation-processing",
        description="Daily processing of new astronomical observations",
        tags=deploy_config.tags + ["observations", "daily"],
        # schedule=CronSchedule(cron=config.observation_processing["schedule_cron"]),
        # TODO: Fix scheduling
        work_queue_name=deploy_config.work_queue_name,
        parameters={
            "max_observations": config.observation_processing[
                "max_observations_per_run"
            ],
        },
    )


def create_model_training_deployment():
    """Create deployment for model training flow."""
    config = get_prefect_config()
    deploy_config = get_deployment_config()

    return train_model_flow.to_deployment(
        name="model-training",
        description="Weekly model training and evaluation",
        tags=deploy_config.tags + ["ml", "training", "weekly"],
        # schedule=CronSchedule(cron=config.model_training["schedule_cron"]),
        # TODO: Fix scheduling
        work_queue_name=deploy_config.work_queue_name,
        parameters={
            "min_f1_score": config.model_training["min_f1_score_threshold"],
        },
    )


def create_retraining_deployment():
    """Create deployment for scheduled retraining flow."""
    config = get_prefect_config()
    deploy_config = get_deployment_config()

    return scheduled_retraining_flow.to_deployment(
        name="scheduled-retraining",
        description="Automated model retraining based on performance and new data",
        tags=deploy_config.tags + ["ml", "retraining", "automated"],
        # schedule=CronSchedule(cron="0 6 * * *"),  # Daily at 6 AM -
        # TODO: Fix scheduling
        work_queue_name=deploy_config.work_queue_name,
        parameters={
            "min_new_samples": config.model_training["min_new_samples_for_retraining"],
            "performance_threshold": config.model_training["min_f1_score_threshold"],
        },
    )


def create_monitoring_deployment():
    """Create deployment for system monitoring flow."""
    deploy_config = get_deployment_config()

    return system_monitoring_flow.to_deployment(
        name="system-monitoring",
        description="Regular system health and performance monitoring",
        tags=deploy_config.tags + ["monitoring", "health"],
        # schedule=CronSchedule(cron=config.monitoring["health_check_schedule_cron"]),
        # TODO: Fix scheduling
        work_queue_name=deploy_config.work_queue_name,
    )


def create_daily_report_deployment():
    """Create deployment for daily report flow."""
    deploy_config = get_deployment_config()

    return daily_report_flow.to_deployment(
        name="daily-report",
        description="Daily operational report generation",
        tags=deploy_config.tags + ["reporting", "daily"],
        # schedule=CronSchedule(cron=config.monitoring["daily_report_cron"]),
        # TODO: Fix scheduling
        work_queue_name=deploy_config.work_queue_name,
    )


def deploy_all_flows():
    """Deploy all AstrID flows to Prefect server."""
    logger.info("Starting deployment of all AstrID flows")

    try:
        # Create all deployments
        deployments = [
            create_observation_processing_deployment(),
            create_model_training_deployment(),
            create_retraining_deployment(),
            create_monitoring_deployment(),
            create_daily_report_deployment(),
        ]

        # Serve all deployments
        logger.info(f"Deploying {len(deployments)} flows")

        # Use serve() without await since it's not async
        serve(*cast(tuple[Any, ...], tuple(deployments)))

        logger.info("All flows deployed successfully")

    except Exception as e:
        logger.error(f"Error deploying flows: {e}")
        raise


def main():
    """Main deployment function."""
    logger.info("AstrID Prefect Flow Deployment")

    try:
        deploy_all_flows()
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
