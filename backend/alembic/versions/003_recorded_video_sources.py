"""add recorded video camera sources

Revision ID: 003_recorded_video_sources
Revises: 002_camera_source_types
"""

from alembic import op
import sqlalchemy as sa


revision = "003_recorded_video_sources"
down_revision = "002_camera_source_types"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("cameras", sa.Column("source_file_path", sa.Text(), nullable=True))
    op.add_column("cameras", sa.Column("source_file_name", sa.String(length=255), nullable=True))
    op.add_column("cameras", sa.Column("source_duration_seconds", sa.Float(), nullable=True))
    op.add_column("cameras", sa.Column("source_fps", sa.Float(), nullable=True))


def downgrade():
    op.drop_column("cameras", "source_fps")
    op.drop_column("cameras", "source_duration_seconds")
    op.drop_column("cameras", "source_file_name")
    op.drop_column("cameras", "source_file_path")
