"""persist recognition runs and diagnostics

Revision ID: 004_recognition_pipeline
Revises: 003_recorded_video_sources
Create Date: 2026-06-30
"""


revision = "004_recognition_pipeline"
down_revision = "003_recorded_video_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Kept as an empty compatibility revision. The object-track schema now
    # lives in the initial migration for clean installs.
    pass


def downgrade() -> None:
    pass
