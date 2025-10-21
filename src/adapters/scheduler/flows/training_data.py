"""Prefect flow for training data collection (ASTR-113)."""

from __future__ import annotations

from prefect import flow, task

from src.adapters.workers.training_data import collect_training_data_worker


@task
def trigger_collection(params: dict) -> None:
    collect_training_data_worker.send(params)


@flow
def training_data_collection_flow(collection_params: dict) -> None:
    trigger_collection.submit(collection_params)
