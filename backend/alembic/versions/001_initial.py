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

    op.create_table('watchlist',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plate_number', sa.String(20), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('alert_channels', postgresql.JSON(), server_default='["frontend"]'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_watchlist_plate_number', 'watchlist', ['plate_number'])

    op.create_table('vehicle_tracks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.Integer(), nullable=False),
        sa.Column('track_id', sa.Integer(), nullable=False),
        sa.Column('vehicle_class', sa.String(50), nullable=False),
        sa.Column('final_plate', sa.String(20), nullable=True),
        sa.Column('plate_status', sa.String(30), server_default='searching'),
        sa.Column('final_plate_confidence', sa.Float(), server_default='0.0'),
        sa.Column('color', sa.String(30), server_default='unknown'),
        sa.Column('color_confidence', sa.Float(), server_default='0.0'),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('duration_seconds', sa.Float(), server_default='0.0'),
        sa.Column('best_vehicle_crop_path', sa.String(500), nullable=True),
        sa.Column('best_plate_crop_path', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_vehicle_tracks_camera_id', 'vehicle_tracks', ['camera_id'])
    op.create_index('ix_vehicle_tracks_final_plate', 'vehicle_tracks', ['final_plate'])

    op.create_table('plate_candidates',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('vehicle_track_id', sa.Integer(), nullable=False),
        sa.Column('plate_text', sa.String(20), nullable=False),
        sa.Column('raw_ocr_text', sa.String(100), nullable=True),
        sa.Column('confidence', sa.Float(), server_default='0.0'),
        sa.Column('regex_valid', sa.Boolean(), server_default='false'),
        sa.Column('regex_score', sa.Float(), server_default='0.0'),
        sa.Column('image_quality_score', sa.Float(), server_default='0.0'),
        sa.Column('frame_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('crop_path', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['vehicle_track_id'], ['vehicle_tracks.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table('events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.Integer(), nullable=False),
        sa.Column('vehicle_track_id', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('payload_json', postgresql.JSON(), nullable=True),
        sa.Column('frame_path', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['vehicle_track_id'], ['vehicle_tracks.id'], ondelete='SET NULL'),
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
    op.drop_table('plate_candidates')
    op.drop_table('vehicle_tracks')
    op.drop_table('watchlist')
    op.drop_table('users')
    op.drop_table('cameras')
