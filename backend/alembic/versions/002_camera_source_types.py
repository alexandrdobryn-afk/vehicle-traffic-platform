"""add camera source types

Revision ID: 002_camera_source_types
Revises: 001_initial
Create Date: 2026-06-29
"""
from alembic import op
import sqlalchemy as sa

revision = "002_camera_source_types"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column("source_type", sa.String(length=20), nullable=False, server_default="rtsp"),
    )
    op.add_column(
        "cameras",
        sa.Column("snapshot_interval_seconds", sa.Float(), nullable=False, server_default="1.0"),
    )


def downgrade() -> None:
    op.drop_column("cameras", "snapshot_interval_seconds")
    op.drop_column("cameras", "source_type")
