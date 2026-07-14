"""training module initial schema

Revision ID: 002_training
Revises: 001_initial
Create Date: 2026-06-25
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '002_training'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('tr_datasets',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('annotation_type', sa.String(50), nullable=False),
        sa.Column('classes', postgresql.JSON(), server_default='[]'),
        sa.Column('status', sa.String(30), server_default='creating'),
        sa.Column('version', sa.String(20), server_default='1.0'),
        sa.Column('image_count', sa.Integer(), server_default='0'),
        sa.Column('video_count', sa.Integer(), server_default='0'),
        sa.Column('annotation_count', sa.Integer(), server_default='0'),
        sa.Column('storage_path', sa.String(500), nullable=True),
        sa.Column('author_id', sa.Integer(), nullable=True),
        sa.Column('author_email', sa.String(255), nullable=True),
        sa.Column('tags', postgresql.JSON(), server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table('tr_dataset_videos',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('dataset_id', sa.Integer(), sa.ForeignKey('tr_datasets.id', ondelete='CASCADE'), nullable=False),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_size', sa.BigInteger(), server_default='0'),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('total_frames', sa.Integer(), nullable=True),
        sa.Column('extracted_frames', sa.Integer(), server_default='0'),
        sa.Column('status', sa.String(30), server_default='uploaded'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table('tr_dataset_images',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('dataset_id', sa.Integer(), sa.ForeignKey('tr_datasets.id', ondelete='CASCADE'), nullable=False),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('original_filename', sa.String(500), nullable=True),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_size', sa.BigInteger(), server_default='0'),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('source', sa.String(50), server_default='upload'),
        sa.Column('source_video_id', sa.Integer(), sa.ForeignKey('tr_dataset_videos.id'), nullable=True),
        sa.Column('source_frame_idx', sa.Integer(), nullable=True),
        sa.Column('split', sa.String(10), nullable=True),
        sa.Column('is_annotated', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_tr_dataset_images_dataset_id', 'tr_dataset_images', ['dataset_id'])
    op.create_index('ix_tr_dataset_images_split', 'tr_dataset_images', ['split'])

    op.create_table('tr_dataset_splits',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('dataset_id', sa.Integer(), sa.ForeignKey('tr_datasets.id', ondelete='CASCADE'), nullable=False),
        sa.Column('split_name', sa.String(20), nullable=False),
        sa.Column('image_count', sa.Integer(), server_default='0'),
        sa.Column('ratio', sa.Float(), server_default='0'),
        sa.Column('seed', sa.Integer(), server_default='42'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table('tr_annotations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('image_id', sa.Integer(), sa.ForeignKey('tr_dataset_images.id', ondelete='CASCADE'), nullable=False),
        sa.Column('annotation_type', sa.String(30), nullable=False),
        sa.Column('class_name', sa.String(100), nullable=True),
        sa.Column('class_id', sa.Integer(), nullable=True),
        sa.Column('x_center', sa.Float(), nullable=True),
        sa.Column('y_center', sa.Float(), nullable=True),
        sa.Column('bbox_width', sa.Float(), nullable=True),
        sa.Column('bbox_height', sa.Float(), nullable=True),
        sa.Column('ocr_text', sa.String(50), nullable=True),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('is_auto', sa.Boolean(), server_default='false'),
        sa.Column('is_verified', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_tr_annotations_image_id', 'tr_annotations', ['image_id'])

    op.create_table('tr_model_versions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('architecture', sa.String(100), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('version_number', sa.Integer(), server_default='1'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('changelog', sa.Text(), nullable=True),
        sa.Column('job_id', sa.Integer(), nullable=True),
        sa.Column('dataset_id', sa.Integer(), nullable=True),
        sa.Column('dataset_name', sa.String(255), nullable=True),
        sa.Column('weights_path', sa.String(1000), nullable=True),
        sa.Column('onnx_path', sa.String(1000), nullable=True),
        sa.Column('trt_path', sa.String(1000), nullable=True),
        sa.Column('metrics', postgresql.JSON(), nullable=True),
        sa.Column('hyperparams', postgresql.JSON(), nullable=True),
        sa.Column('deploy_status', sa.String(30), server_default='pending'),
        sa.Column('is_production', sa.Boolean(), server_default='false'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('validation_passed', sa.Boolean(), nullable=True),
        sa.Column('auto_test_results', postgresql.JSON(), nullable=True),
        sa.Column('benchmark_vs_prev', postgresql.JSON(), nullable=True),
        sa.Column('author_id', sa.Integer(), nullable=True),
        sa.Column('author_email', sa.String(255), nullable=True),
        sa.Column('approved_by', sa.String(255), nullable=True),
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table('tr_jobs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('architecture', sa.String(100), nullable=False),
        sa.Column('dataset_id', sa.Integer(), sa.ForeignKey('tr_datasets.id'), nullable=False),
        sa.Column('status', sa.String(30), server_default='queued'),
        sa.Column('celery_task_id', sa.String(255), nullable=True),
        sa.Column('hyperparams', postgresql.JSON(), server_default='{}'),
        sa.Column('augmentation_config', postgresql.JSON(), server_default='{}'),
        sa.Column('current_epoch', sa.Integer(), server_default='0'),
        sa.Column('total_epochs', sa.Integer(), server_default='100'),
        sa.Column('progress_pct', sa.Float(), server_default='0'),
        sa.Column('eta_seconds', sa.Integer(), nullable=True),
        sa.Column('best_metrics', postgresql.JSON(), nullable=True),
        sa.Column('final_metrics', postgresql.JSON(), nullable=True),
        sa.Column('output_model_id', sa.Integer(), nullable=True),
        sa.Column('gpu_device', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('log_path', sa.String(500), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('author_id', sa.Integer(), nullable=True),
        sa.Column('author_email', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    op.create_table('tr_metrics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('job_id', sa.Integer(), sa.ForeignKey('tr_jobs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('epoch', sa.Integer(), nullable=False),
        sa.Column('train_loss', sa.Float(), nullable=True),
        sa.Column('val_loss', sa.Float(), nullable=True),
        sa.Column('precision', sa.Float(), nullable=True),
        sa.Column('recall', sa.Float(), nullable=True),
        sa.Column('map50', sa.Float(), nullable=True),
        sa.Column('map50_95', sa.Float(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('char_accuracy', sa.Float(), nullable=True),
        sa.Column('text_accuracy', sa.Float(), nullable=True),
        sa.Column('gpu_memory_mb', sa.Float(), nullable=True),
        sa.Column('gpu_utilization', sa.Float(), nullable=True),
        sa.Column('lr', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_tr_metrics_job_id', 'tr_metrics', ['job_id'])

    op.create_table('tr_deploy_logs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_version_id', sa.Integer(), sa.ForeignKey('tr_model_versions.id', ondelete='CASCADE')),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('actor_email', sa.String(255), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('previous_model_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )


def downgrade() -> None:
    for t in ['tr_deploy_logs', 'tr_metrics', 'tr_jobs', 'tr_model_versions',
              'tr_annotations', 'tr_dataset_splits', 'tr_dataset_images',
              'tr_dataset_videos', 'tr_datasets']:
        op.drop_table(t)
