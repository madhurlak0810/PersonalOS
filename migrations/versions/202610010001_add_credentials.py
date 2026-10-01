"""Add credentials: references to secrets held in the OS secret store.

A row says a credential exists, whose it is, which provider it is for and what
it may be used for. It has no column that holds the credential. The refresh
token or API key lives in the OS keychain under `credential_ref`
(`cred://<provider>/<name>`), and only the executor's credential broker
exchanges that reference for a short-lived access token, at execution time.

The check constraint on `credential_ref` is a tripwire: the application-level
guarantee is that `CredentialRepository.create` accepts a `CredentialRef` and
nothing else, and this stops a row written around the repository from putting
an arbitrary string in the one column shaped like it could take one.

Revision ID: 202610010001
Revises: 202609290002
Create Date: 2026-10-01

"""

from alembic import op
import sqlalchemy as sa

from personalos.persistence.models import GUID

# revision identifiers, used by Alembic.
revision = "202610010001"
down_revision = "202609290002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "credentials",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("provider", sa.String(length=63), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "oauth_refresh_token",
                "api_key",
                "oauth_client_secret",
                name="credential_kind",
            ),
            nullable=False,
        ),
        sa.Column("credential_ref", sa.String(length=300), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("active", "revoked", name="credential_status"),
            nullable=False,
            server_default="active",
        ),
        sa.Column("last_exchanged_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        # Inline for the same reason as in `pending_checkpoints`: SQLite cannot
        # add a constraint to an existing table.
        sa.UniqueConstraint("credential_ref", name="uq_credentials_credential_ref"),
        sa.CheckConstraint(
            "substr(credential_ref, 1, 7) = 'cred://'", name="ck_credentials_ref_is_reference"
        ),
    )
    op.create_index("ix_credentials_user_id", "credentials", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_credentials_user_id", table_name="credentials")
    op.drop_table("credentials")
    bind = op.get_bind()
    sa.Enum(name="credential_status").drop(bind, checkfirst=True)
    sa.Enum(name="credential_kind").drop(bind, checkfirst=True)
