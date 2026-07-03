"""Docker-backed control-plane lifecycle test for the Training API.

Run inside the Training API container while PostgreSQL, Redis, and a Celery
worker are available:

    python -m unittest tests.test_training_api_integration -v

The test exercises the real API and PostgreSQL schema. It mocks only the
heavy Celery training dispatch and inference artifact copy.
"""

from __future__ import annotations

import base64
import shutil
import uuid
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.orm import configure_mappers, sessionmaker

from app.config import settings
from app.main import app
from app.models.database import (
    Dataset,
    DeployLog,
    ModelVersion,
    TrainingJob,
)
from app.services.model_registry_service import model_registry_service


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)

sync_engine = create_engine(settings.DATABASE_SYNC_URL)
SyncSession = sessionmaker(bind=sync_engine, expire_on_commit=False)


def auth_headers() -> dict[str, str]:
    token = jwt.encode(
        {"sub": "1", "role": "admin", "email": "training-integration@local"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}


class TrainingApiIntegrationTest(TestCase):
    dataset_id: int | None = None
    dataset_path: str | None = None
    job_id: int | None = None
    model_ids: list[int] = []

    @classmethod
    def setUpClass(cls):
        """Remove artifacts left by an interrupted previous integration run."""
        storage_paths: list[str] = []
        with SyncSession() as db:
            datasets = db.execute(
                select(Dataset.id, Dataset.storage_path).where(
                    Dataset.name.like("integration-%")
                )
            ).all()
            dataset_ids = [row.id for row in datasets]
            storage_paths = [row.storage_path for row in datasets if row.storage_path]

            if dataset_ids:
                model_ids = list(
                    db.scalars(
                        select(ModelVersion.id).where(
                            ModelVersion.dataset_id.in_(dataset_ids)
                        )
                    )
                )
                db.execute(
                    update(TrainingJob)
                    .where(TrainingJob.dataset_id.in_(dataset_ids))
                    .values(output_model_id=None)
                )
                if model_ids:
                    db.execute(
                        delete(DeployLog).where(
                            DeployLog.model_version_id.in_(model_ids)
                        )
                    )
                    db.execute(
                        delete(ModelVersion).where(ModelVersion.id.in_(model_ids))
                    )
                db.execute(
                    delete(TrainingJob).where(
                        TrainingJob.dataset_id.in_(dataset_ids)
                    )
                )
                db.execute(delete(Dataset).where(Dataset.id.in_(dataset_ids)))
                db.commit()

        for storage_path in storage_paths:
            shutil.rmtree(storage_path, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        cls._cleanup()
        sync_engine.dispose()

    @classmethod
    def _cleanup(cls):
        with SyncSession() as db:
            if cls.job_id:
                db.execute(
                    update(TrainingJob)
                    .where(TrainingJob.id == cls.job_id)
                    .values(output_model_id=None)
                )
            if cls.model_ids:
                db.execute(delete(DeployLog).where(DeployLog.model_version_id.in_(cls.model_ids)))
                db.execute(delete(ModelVersion).where(ModelVersion.id.in_(cls.model_ids)))
            if cls.job_id:
                db.execute(delete(TrainingJob).where(TrainingJob.id == cls.job_id))
            if cls.dataset_id:
                dataset = db.get(Dataset, cls.dataset_id)
                if dataset:
                    db.delete(dataset)
            db.commit()
        if cls.dataset_path:
            shutil.rmtree(cls.dataset_path, ignore_errors=True)

    @staticmethod
    def _create_model_versions(job_id: int, dataset_id: int, suffix: str) -> list[int]:
        with SyncSession() as db:
            models = []
            for number in (1, 2):
                model = ModelVersion(
                    name=f"integration_plate_model_{suffix}_{number}",
                    model_type="plate_detector",
                    architecture="yolov8n",
                    version=f"v{number}.0",
                    version_number=number,
                    job_id=job_id,
                    dataset_id=dataset_id,
                    dataset_name=f"integration-{suffix}",
                    weights_path=f"/tmp/integration-{suffix}-{number}.pt",
                    metrics={"map50": 0.5 + number / 10},
                    artifact_metadata={"source": "integration_test"},
                    deploy_status="pending",
                    validation_passed=True,
                    auto_test_results={"passed": True},
                    author_email="training-integration@local",
                )
                db.add(model)
                models.append(model)
            db.commit()
            for model in models:
                db.refresh(model)
            return [model.id for model in models]

    def test_control_plane_lifecycle(self):
        configure_mappers()
        suffix = uuid.uuid4().hex[:10]
        headers = auth_headers()

        with TestClient(app) as client:
            health = client.get("/api/v1/training/health")
            self.assertEqual(health.status_code, 200, health.text)
            self.assertEqual(health.json()["orm"], "ok")
            self.assertEqual(health.json()["database"], "ok")

            for endpoint in ("datasets", "jobs", "registry"):
                response = client.get(f"/api/v1/training/{endpoint}", headers=headers)
                self.assertEqual(response.status_code, 200, response.text)

            created = client.post(
                "/api/v1/training/datasets",
                headers=headers,
                json={
                    "name": f"integration-{suffix}",
                    "description": "Small lifecycle dataset; not a quality benchmark",
                    "model_type": "plate_detector",
                    "annotation_type": "bbox",
                    "classes": ["license_plate"],
                    "tags": ["integration-test"],
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            dataset = created.json()
            type(self).dataset_id = dataset["id"]

            detail = client.get(
                f"/api/v1/training/datasets/{self.dataset_id}", headers=headers
            )
            self.assertEqual(detail.status_code, 200, detail.text)

            files = [
                ("files", (f"sample-{index}.png", PNG_1X1, "image/png"))
                for index in range(1, 5)
            ]
            uploaded = client.post(
                f"/api/v1/training/datasets/{self.dataset_id}/images",
                headers=headers,
                files=files,
            )
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            self.assertEqual(uploaded.json()["uploaded"], 4)

            images = client.get(
                f"/api/v1/training/datasets/{self.dataset_id}/images",
                headers=headers,
            ).json()
            for image in images:
                annotation = client.post(
                    f"/api/v1/training/annotations/image/{image['id']}",
                    headers=headers,
                    json={
                        "annotation_type": "bbox",
                        "class_name": "license_plate",
                        "class_id": 0,
                        "x_center": 0.5,
                        "y_center": 0.5,
                        "bbox_width": 0.5,
                        "bbox_height": 0.25,
                    },
                )
                self.assertEqual(annotation.status_code, 200, annotation.text)

            split = client.post(
                f"/api/v1/training/datasets/{self.dataset_id}/split",
                headers=headers,
                json={"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "seed": 42},
            )
            self.assertEqual(split.status_code, 200, split.text)
            self.assertEqual(sum(split.json()["counts"].values()), 4)

            mismatch = client.post(
                "/api/v1/training/jobs",
                headers=headers,
                json={
                    "name": f"mismatch-{suffix}",
                    "model_type": "vehicle_detector",
                    "architecture": "yolo11n",
                    "dataset_id": self.dataset_id,
                },
            )
            self.assertEqual(mismatch.status_code, 400, mismatch.text)

            invalid_arch = client.post(
                "/api/v1/training/jobs",
                headers=headers,
                json={
                    "name": f"invalid-arch-{suffix}",
                    "model_type": "plate_detector",
                    "architecture": "not-a-model",
                    "dataset_id": self.dataset_id,
                },
            )
            self.assertEqual(invalid_arch.status_code, 400, invalid_arch.text)

            with patch("app.celery_app.celery_app.send_task") as send_task:
                send_task.return_value = SimpleNamespace(id=f"mock-task-{suffix}")
                job_response = client.post(
                    "/api/v1/training/jobs",
                    headers=headers,
                    json={
                        "name": f"plate-smoke-{suffix}",
                        "model_type": "plate_detector",
                        "architecture": "yolov8n",
                        "dataset_id": self.dataset_id,
                        "hyperparams": {"epochs": 1, "batch_size": 1, "workers": 0},
                    },
                )
            self.assertEqual(job_response.status_code, 200, job_response.text)
            type(self).job_id = job_response.json()["id"]
            self.assertEqual(job_response.json()["celery_task_id"], f"mock-task-{suffix}")

            job_list = client.get("/api/v1/training/jobs", headers=headers)
            self.assertEqual(job_list.status_code, 200, job_list.text)
            self.assertTrue(any(job["id"] == self.job_id for job in job_list.json()))

            type(self).model_ids = self._create_model_versions(
                self.job_id, self.dataset_id, suffix
            )
            registry = client.get("/api/v1/training/registry", headers=headers)
            self.assertEqual(registry.status_code, 200, registry.text)
            self.assertTrue(set(self.model_ids).issubset({model["id"] for model in registry.json()}))

            copy_result = {"success": True, "inference_path": "/mock/production.pt"}
            with patch.object(
                model_registry_service,
                "_copy_to_inference",
                new=AsyncMock(return_value=copy_result),
            ):
                for model_id in self.model_ids:
                    approved = client.post(
                        f"/api/v1/training/registry/{model_id}/approve",
                        headers=headers,
                        json={"comment": "integration approval", "run_auto_tests": True},
                    )
                    self.assertEqual(approved.status_code, 200, approved.text)
                    self.assertTrue(approved.json()["success"])

                rolled_back = client.post(
                    "/api/v1/training/registry/rollback",
                    headers=headers,
                    json={"model_version_id": self.model_ids[0], "comment": "integration rollback"},
                )
                self.assertEqual(rolled_back.status_code, 200, rolled_back.text)
                self.assertTrue(rolled_back.json()["success"])

            rejected = client.post(
                f"/api/v1/training/registry/{self.model_ids[1]}/reject",
                headers=headers,
                params={"comment": "integration rejection"},
            )
            self.assertEqual(rejected.status_code, 200, rejected.text)
            self.assertTrue(rejected.json()["success"])

            with SyncSession() as db:
                stored = db.get(Dataset, self.dataset_id)
                type(self).dataset_path = stored.storage_path if stored else None
