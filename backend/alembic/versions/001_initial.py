"""initial schema

Revision ID: 001_initial
Revises:
Create Date: 2026-06-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('cameras',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('rtsp_url_encrypted', sa.Text(), nullable=False),
        sa.Column('location', sa.String(500), nullable=True),
        sa.Column('status', sa.String(50), server_default='offline'),
        sa.Column('ai_mode', sa.String(50), server_default='balanced'),
        sa.Column('priority', sa.Integer(), server_default='1'),
        sa.Column('max_fps', sa.Integer(), server_default='25'),
        sa.Column('is_active', sa.Boolean(), server_default='false'),
        sa.Column('save_crops', sa.Boolean(), server_default='true'),
        sa.Column('anonymization', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('role', sa.String(20), server_default='viewer'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index('ix_users_email', 'users', ['email'])

    op.create_table('object_tracks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('processing_run_id', sa.String(64), nullable=False, server_default='default'),
        sa.Column('track_id', sa.Integer(), nullable=False),
        sa.Column('object_class', sa.String(100), nullable=False),
        sa.Column('confidence', sa.Float(), server_default='0.0'),
        sa.Column('trajectory', postgresql.JSON(), server_default='[]'),
        sa.Column('speed_pixels_per_second', sa.Float(), server_default='0.0'),
        sa.Column('direction_degrees', sa.Float(), nullable=True),
        sa.Column('state', sa.String(30), server_default='active'),
        sa.Column('attributes', postgresql.JSON(), server_default='{}'),
        sa.Column('best_crop_path', sa.String(500), nullable=True),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('duration_seconds', sa.Float(), server_default='0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['source_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_object_tracks_source_id', 'object_tracks', ['source_id'])
    op.create_index('ix_object_tracks_object_class', 'object_tracks', ['object_class'])
    op.create_index('ix_object_tracks_processing_run', 'object_tracks', ['source_id', 'processing_run_id'])

    op.create_table('events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.Integer(), nullable=False),
        sa.Column('object_track_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('payload_json', postgresql.JSON(), nullable=True),
        sa.Column('frame_path', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['object_track_id'], ['object_tracks.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_events_camera_id', 'events', ['camera_id'])
    op.create_index('ix_events_event_type', 'events', ['event_type'])
    op.create_index('ix_events_created_at', 'events', ['created_at'])

    op.create_table('system_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('level', sa.String(20), nullable=False),
        sa.Column('source', sa.String(100), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('payload_json', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table('app_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(100), nullable=False),
        sa.Column('value', postgresql.JSON(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )


def downgrade() -> None:
    op.drop_table('app_settings')
    op.drop_table('system_logs')
    op.drop_table('events')
    op.drop_table('object_tracks')
    op.drop_table('users')
    op.drop_table('cameras')
