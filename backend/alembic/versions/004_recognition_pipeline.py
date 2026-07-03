"""persist recognition runs and diagnostics

Revision ID: 004_recognition_pipeline
Revises: 003_recorded_video_sources
Create Date: 2026-06-30
"""
from alembic import op
import sqlalchemy as sa


revision = "004_recognition_pipeline"
down_revision = "003_recorded_video_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "vehicle_tracks",
        sa.Column("processing_run_id", sa.String(64), nullable=False, server_default="legacy"),
    )
    op.add_column(
        "vehicle_tracks",
        sa.Column("recognition_diagnostics", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_vehicle_tracks_processing_run",
        "vehicle_tracks",
        ["camera_id", "processing_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_vehicle_tracks_processing_run", table_name="vehicle_tracks")
    op.drop_column("vehicle_tracks", "recognition_diagnostics")
    op.drop_column("vehicle_tracks", "processing_run_id")
