"""add object registry evidence fields

Revision ID: 006_object_registry_fields
Revises: 005_pipeline_modes
Create Date: 2026-07-12
"""

from alembic import op
import sqlalchemy as sa


revision = "006_object_registry_fields"
down_revision = "005_pipeline_modes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("object_tracks", sa.Column("last_bbox", sa.JSON(), nullable=False, server_default="[]"))
    op.add_column("object_tracks", sa.Column("first_video_timestamp_seconds", sa.Float(), nullable=True))
    op.add_column("object_tracks", sa.Column("last_video_timestamp_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("object_tracks", "last_video_timestamp_seconds")
    op.drop_column("object_tracks", "first_video_timestamp_seconds")
    op.drop_column("object_tracks", "last_bbox")
