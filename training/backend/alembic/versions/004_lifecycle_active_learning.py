"""Add review lifecycle, active learning, and evaluation reports.

Revision ID: 004_lifecycle_active_learning
Revises: 003_model_provenance
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "004_lifecycle_active_learning"
down_revision = "003_model_provenance"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("tr_dataset_images", sa.Column("frame_status", sa.String(30), nullable=False, server_default="unlabeled"))
    op.add_column("tr_dataset_images", sa.Column("review_priority", sa.Float(), nullable=False, server_default="0"))
    op.add_column("tr_dataset_images", sa.Column("review_reason", sa.String(100), nullable=True))
    op.add_column("tr_dataset_images", sa.Column("scene_tags", postgresql.JSON(), nullable=False, server_default="[]"))
    op.add_column("tr_dataset_images", sa.Column("quality_tags", postgresql.JSON(), nullable=False, server_default="[]"))
    op.add_column("tr_dataset_images", sa.Column("frame_metadata", postgresql.JSON(), nullable=False, server_default="{}"))

    op.add_column("tr_jobs", sa.Column("training_mode", sa.String(50), nullable=False, server_default="full_finetune"))
    op.add_column("tr_jobs", sa.Column("tile_config", postgresql.JSON(), nullable=False, server_default="{}"))
    op.add_column("tr_jobs", sa.Column("evaluation_policy", postgresql.JSON(), nullable=False, server_default="{}"))

    op.add_column("tr_model_versions", sa.Column("gate_result", sa.String(30), nullable=True))
    op.add_column("tr_model_versions", sa.Column("gate_reasons", postgresql.JSON(), nullable=False, server_default="[]"))
    op.add_column("tr_model_versions", sa.Column("evaluation_report_id", sa.Integer(), nullable=True))

    op.create_table(
        "tr_evaluation_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("tr_model_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("tr_datasets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("job_id", sa.Integer(), sa.ForeignKey("tr_jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("summary", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("per_class_metrics", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("slice_metrics", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("speed_metrics", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("confusion_matrix", postgresql.JSON(), nullable=True),
        sa.Column("gate_result", sa.String(30), nullable=False, server_default="candidate"),
        sa.Column("gate_reasons", postgresql.JSON(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_tr_evaluation_reports_model_version_id", "tr_evaluation_reports", ["model_version_id"])
    op.create_index("ix_tr_evaluation_reports_dataset_id", "tr_evaluation_reports", ["dataset_id"])
    op.create_index("ix_tr_evaluation_reports_job_id", "tr_evaluation_reports", ["job_id"])

    op.create_table(
        "tr_evaluation_errors",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("report_id", sa.Integer(), sa.ForeignKey("tr_evaluation_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_image_id", sa.Integer(), sa.ForeignKey("tr_dataset_images.id", ondelete="SET NULL"), nullable=True),
        sa.Column("error_type", sa.String(50), nullable=False),
        sa.Column("class_name", sa.String(100), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("bbox", postgresql.JSON(), nullable=True),
        sa.Column("details", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_tr_evaluation_errors_report_id", "tr_evaluation_errors", ["report_id"])
    op.create_index("ix_tr_evaluation_errors_dataset_image_id", "tr_evaluation_errors", ["dataset_image_id"])

    op.create_table(
        "tr_active_learning_items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("tr_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_id", sa.Integer(), sa.ForeignKey("tr_dataset_images.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_version_id", sa.Integer(), sa.ForeignKey("tr_model_versions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("evaluation_error_id", sa.Integer(), sa.ForeignKey("tr_evaluation_errors.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reason", sa.String(80), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(30), nullable=False, server_default="open"),
        sa.Column("suggested_class", sa.String(100), nullable=True),
        sa.Column("source", sa.String(80), nullable=False, server_default="manual"),
        sa.Column("details", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("reviewer_email", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_tr_active_learning_items_dataset_id", "tr_active_learning_items", ["dataset_id"])
    op.create_index("ix_tr_active_learning_items_image_id", "tr_active_learning_items", ["image_id"])
    op.create_index("ix_tr_active_learning_items_model_version_id", "tr_active_learning_items", ["model_version_id"])
    op.create_index("ix_tr_active_learning_items_evaluation_error_id", "tr_active_learning_items", ["evaluation_error_id"])
    op.create_index("ix_tr_active_learning_items_status", "tr_active_learning_items", ["status"])


def downgrade():
    op.drop_table("tr_active_learning_items")
    op.drop_table("tr_evaluation_errors")
    op.drop_table("tr_evaluation_reports")

    op.drop_column("tr_model_versions", "evaluation_report_id")
    op.drop_column("tr_model_versions", "gate_reasons")
    op.drop_column("tr_model_versions", "gate_result")
    op.drop_column("tr_jobs", "evaluation_policy")
    op.drop_column("tr_jobs", "tile_config")
    op.drop_column("tr_jobs", "training_mode")
    op.drop_column("tr_dataset_images", "frame_metadata")
    op.drop_column("tr_dataset_images", "quality_tags")
    op.drop_column("tr_dataset_images", "scene_tags")
    op.drop_column("tr_dataset_images", "review_reason")
    op.drop_column("tr_dataset_images", "review_priority")
    op.drop_column("tr_dataset_images", "frame_status")
