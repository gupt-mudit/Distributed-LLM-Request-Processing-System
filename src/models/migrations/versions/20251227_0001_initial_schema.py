"""initial schema

Revision ID: 20251227_0001
Revises:
Create Date: 2025-12-27 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from src.models.prompt_request import PromptPriority, PromptStatus

# revision identifiers, used by Alembic.
revision = "20251227_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "prompt_cache_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column(
            "hit_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.Column("last_hit_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "prompt_requests",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                PromptStatus,
                name="prompt_status",
                values_callable=lambda enum: [member.value for member in enum],
            ),
            server_default=PromptStatus.RECEIVED.value,
            nullable=False,
        ),
        sa.Column(
            "priority",
            sa.Enum(
                PromptPriority,
                name="prompt_priority",
                values_callable=lambda enum: [member.value for member in enum],
            ),
            server_default=PromptPriority.NORMAL.value,
            nullable=False,
        ),
        sa.Column(
            "cached",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "retry_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.String(length=255), nullable=True),
        sa.Column(
            "cache_entry_id",
            sa.Integer(),
            sa.ForeignKey("prompt_cache_entries.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("timezone('utc', now())"),
            onupdate=sa.text("timezone('utc', now())"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "user_id",
            "prompt_id",
            name="uq_prompt_requests_user_prompt",
        ),
    )

    op.create_index(
        "ix_prompt_cache_entries_created_at",
        "prompt_cache_entries",
        ["created_at"],
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_prompt_cache_entries_embedding
        ON prompt_cache_entries
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_prompt_cache_entries_embedding")
    op.drop_index(
        "ix_prompt_cache_entries_created_at",
        table_name="prompt_cache_entries",
    )
    op.drop_table("prompt_requests")
    op.drop_table("prompt_cache_entries")
    op.execute("DROP TYPE IF EXISTS prompt_status")
    op.execute("DROP TYPE IF EXISTS prompt_priority")

