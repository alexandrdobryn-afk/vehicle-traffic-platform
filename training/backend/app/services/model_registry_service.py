import os
import shutil
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.database import ModelVersion, DeployLog, TrainingJob
from app.schemas.schemas import DeployApproval, RollbackRequest
from app.config import settings

logger = logging.getLogger(__name__)


class ModelRegistryService:

    async def list_versions(
        self,
        db: AsyncSession,
        model_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[ModelVersion]:
        q = select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(limit)
        if model_type:
            q = q.where(ModelVersion.model_type == model_type)
        result = await db.execute(q)
        return result.scalars().all()

    async def get_version(self, db: AsyncSession, mv_id: int) -> Optional[ModelVersion]:
        result = await db.execute(select(ModelVersion).where(ModelVersion.id == mv_id))
        return result.scalar_one_or_none()

    async def get_production(
        self, db: AsyncSession, model_type: str
    ) -> Optional[ModelVersion]:
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.model_type == model_type,
                ModelVersion.is_production == True,
            )
        )
        return result.scalar_one_or_none()

    async def approve_and_deploy(
        self,
        db: AsyncSession,
        mv_id: int,
        approval: DeployApproval,
        actor_email: str,
    ) -> dict:
        """
        Manual approval flow:
        1. Run auto tests if requested
        2. Copy weights to inference backend
        3. Mark as production
        4. Deactivate previous production model
        """
        mv = await self.get_version(db, mv_id)
        if not mv:
            return {"success": False, "error": "Model not found"}

        if mv.deploy_status == "deployed":
            return {"success": False, "error": "Already deployed"}

        if mv.validation_passed is not True:
            return {
                "success": False,
                "error": "Model must pass dataset validation before deployment",
            }
        if mv.gate_result == "rejected":
            return {
                "success": False,
                "error": "Model was rejected by the decision gate",
                "gate_reasons": mv.gate_reasons or [],
            }
        if mv.gate_result == "candidate":
            return {
                "success": False,
                "error": "Candidate model needs stronger evaluation before production deployment",
                "gate_reasons": mv.gate_reasons or [],
            }

        # Auto tests before deploy
        if approval.run_auto_tests:
            # Heavy model checks run in the validation worker. Re-running them
            # inside the lightweight API process would require the whole GPU
            # stack and can deploy a different result than the validated one.
            test_results = mv.auto_test_results or {}
            if test_results.get("passed") is not True:
                mv.deploy_status = "rejected"
                await db.commit()
                return {
                    "success": False,
                    "error": "Stored validation auto tests are missing or failed",
                    "test_results": test_results,
                }

        # Copy weights to inference backend
        deploy_result = await self._copy_to_inference(mv)
        if not deploy_result["success"]:
            return deploy_result

        # Previous production → mark inactive
        old_prod = await self.get_production(db, mv.model_type)
        prev_id = None
        if old_prod and old_prod.id != mv.id:
            old_prod.is_production = False
            old_prod.deploy_status = "rolled_back"
            prev_id = old_prod.id

        # Mark new version as production
        mv.is_production = True
        mv.deploy_status = "deployed"
        mv.approved_by = actor_email
        mv.deployed_at = datetime.now(timezone.utc)

        # Log
        log = DeployLog(
            model_version_id=mv.id,
            action="deployed",
            actor_email=actor_email,
            comment=approval.comment,
            previous_model_id=prev_id,
        )
        db.add(log)
        await db.commit()

        logger.info(f"Model {mv.name} deployed by {actor_email}")
        return {
            "success": True,
            "model_id": mv.id,
            "name": mv.name,
            "inference_path": deploy_result.get("inference_path"),
        }

    async def reject(
        self,
        db: AsyncSession,
        mv_id: int,
        comment: Optional[str],
        actor_email: str,
    ) -> dict:
        mv = await self.get_version(db, mv_id)
        if not mv:
            return {"success": False, "error": "Not found"}
        mv.deploy_status = "rejected"
        log = DeployLog(
            model_version_id=mv.id,
            action="rejected",
            actor_email=actor_email,
            comment=comment,
        )
        db.add(log)
        await db.commit()
        return {"success": True}

    async def rollback(
        self,
        db: AsyncSession,
        request: RollbackRequest,
        actor_email: str,
    ) -> dict:
        """Roll back to a specific previous model version."""
        target = await self.get_version(db, request.model_version_id)
        if not target:
            return {"success": False, "error": "Target model not found"}

        # Deactivate current production
        current = await self.get_production(db, target.model_type)
        prev_id = None
        if current:
            current.is_production = False
            current.deploy_status = "rolled_back"
            prev_id = current.id

        # Copy target weights to inference
        deploy_result = await self._copy_to_inference(target)
        if not deploy_result["success"]:
            if current:
                current.is_production = True
                current.deploy_status = "deployed"
            await db.commit()
            return deploy_result

        # Activate target
        target.is_production = True
        target.deploy_status = "deployed"
        target.deployed_at = datetime.now(timezone.utc)

        log = DeployLog(
            model_version_id=target.id,
            action="rolled_back",
            actor_email=actor_email,
            comment=request.comment,
            previous_model_id=prev_id,
        )
        db.add(log)
        await db.commit()

        logger.info(f"Rolled back to {target.name} by {actor_email}")
        return {
            "success": True,
            "rolled_back_to": target.name,
            "previous_model_id": prev_id,
        }

    async def _copy_to_inference(self, mv: ModelVersion) -> dict:
        """Copy model weights to inference backend models directory."""
        if not mv.weights_path or not os.path.exists(mv.weights_path):
            return {"success": False, "error": "Weights file not found"}

        # Determine destination in inference models dir
        type_to_dir = {
            "object_detector": "object_detector",
            "object_segmenter": "object_segmenter",
            "object_classifier": "object_classifier",
        }
        subdir = type_to_dir.get(mv.model_type, mv.model_type)
        dest_dir = Path(settings.INFERENCE_MODELS_PATH) / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Stable canonical names are what the inference registry reads.
        ext = Path(mv.weights_path).suffix
        dest_file = dest_dir / f"production{ext}"
        self._atomic_copy(Path(mv.weights_path), dest_file)

        # Also copy ONNX if available
        if mv.onnx_path and os.path.exists(mv.onnx_path):
            onnx_dest = dest_dir / "production.onnx"
            self._atomic_copy(Path(mv.onnx_path), onnx_dest)
        if mv.model_type == "object_classifier":
            labels = (mv.artifact_metadata or {}).get("classes") or []
            labels_path = dest_dir / "labels.json"
            temp_labels = labels_path.with_suffix(".json.tmp")
            temp_labels.write_text(json.dumps({"classes": labels}, indent=2) + "\n", encoding="utf-8")
            os.replace(temp_labels, labels_path)

        # Copy TRT if available
        if mv.trt_path and os.path.exists(mv.trt_path):
            trt_dest = dest_dir / "production.engine"
            self._atomic_copy(Path(mv.trt_path), trt_dest)

        metadata = {
            "model_version_id": mv.id,
            "name": mv.name,
            "model_type": mv.model_type,
            "architecture": mv.architecture,
            "version": mv.version,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {},
        }
        for artifact_path in [dest_file, dest_dir / "production.onnx", dest_dir / "production.engine"]:
            if artifact_path.exists():
                metadata["artifacts"][artifact_path.name] = {
                    "sha256": self._sha256(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                }
        metadata_path = dest_dir / "production.json"
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_metadata, metadata_path)

        logger.info(f"Copied {mv.name} to inference: {dest_file}")
        return {"success": True, "inference_path": str(dest_file)}

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        temp = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temp)
        os.replace(temp, destination)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    async def trigger_export(
        self, db: AsyncSession, mv_id: int, format: str
    ) -> dict:
        """Queue ONNX or TRT export task."""
        from app.celery_app import celery_app
        mv = await self.get_version(db, mv_id)
        if not mv:
            return {"success": False, "error": "Not found"}

        if format == "onnx":
            task = celery_app.send_task(
                "app.workers.export_worker.export_onnx",
                kwargs={"model_version_id": mv_id},
                queue="gpu",
            )
        elif format == "tensorrt":
            task = celery_app.send_task(
                "app.workers.export_worker.export_tensorrt",
                kwargs={"model_version_id": mv_id, "half": True},
                queue="gpu",
            )
        else:
            return {"success": False, "error": f"Unknown format: {format}"}

        return {"success": True, "task_id": task.id, "format": format}

    async def trigger_validation(self, db: AsyncSession, mv_id: int) -> dict:
        from app.celery_app import celery_app
        task = celery_app.send_task(
            "app.workers.validation_worker.run_validation",
            kwargs={"model_version_id": mv_id},
            queue="gpu",
        )
        return {"success": True, "task_id": task.id}

    async def get_deploy_logs(
        self, db: AsyncSession, mv_id: int
    ) -> List[DeployLog]:
        result = await db.execute(
            select(DeployLog)
            .where(DeployLog.model_version_id == mv_id)
            .order_by(DeployLog.created_at.desc())
        )
        return result.scalars().all()


model_registry_service = ModelRegistryService()
