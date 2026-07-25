"""Initial schema migration for Users and Predictions tables

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-07-22 23:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum type for prediction labels
    prediction_label_enum = postgresql.ENUM('PNEUMONIA', 'NORMAL', name='prediction_label')
    prediction_label_enum.create(op.get_bind(), checkfirst=True)

    # Users Table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(length=320), nullable=False),
        sa.Column('hashed_password', sa.String(length=128), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # Predictions Table
    op.create_table(
        'predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('original_image_path', sa.String(length=500), nullable=False),
        sa.Column('prediction', sa.Enum('PNEUMONIA', 'NORMAL', name='prediction_label'), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('processing_time_ms', sa.Float(), nullable=False),
        sa.Column('heatmap_path', sa.String(length=500), nullable=True),
        sa.Column('gradcam_observation', sa.Text(), nullable=True),
        sa.Column('report_text', sa.Text(), nullable=True),
        sa.Column('patient_age', sa.Integer(), nullable=True),
        sa.Column('patient_gender', sa.String(length=20), nullable=True),
        sa.Column('patient_symptoms', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f('ix_predictions_user_id'), 'predictions', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_predictions_user_id'), table_name='predictions')
    op.drop_table('predictions')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    
    prediction_label_enum = postgresql.ENUM('PNEUMONIA', 'NORMAL', name='prediction_label')
    prediction_label_enum.drop(op.get_bind(), checkfirst=True)
