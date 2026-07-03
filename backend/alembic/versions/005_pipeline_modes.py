"""add per-source pipeline mode configuration

Revision ID: 005_pipeline_modes
Revises: 004_recognition_pipeline
Create Date: 2026-07-01
"""
from alembic import op
import sqlalchemy as sa


revision = "005_pipeline_modes"
down_revision = "004_recognition_pipeline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cameras",
        sa.Column(
            "pipeline_mode",
            sa.String(20),
            nullable=False,
            server_default="automatic",
        ),
    )
    op.add_column(
        "cameras",
        sa.Column("pipeline_config", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cameras", "pipeline_config")
    op.drop_column("cameras", "pipeline_mode")
