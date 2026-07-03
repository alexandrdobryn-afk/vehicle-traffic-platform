"""Add model artifact provenance metadata.

Revision ID: 003_model_provenance
Revises: 002_training
"""

from alembic import op
import sqlalchemy as sa


revision = "003_model_provenance"
down_revision = "002_training"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("tr_model_versions", sa.Column("artifact_metadata", sa.JSON(), nullable=True))


def downgrade():
    op.drop_column("tr_model_versions", "artifact_metadata")
