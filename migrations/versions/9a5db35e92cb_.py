"""empty message

Revision ID: 9a5db35e92cb
Revises: 9a69371375a5
Create Date: 2021-03-13 18:57:54.746455

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9a5db35e92cb'
down_revision = '9a69371375a5'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('user', 'password_hash')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('user', sa.Column('password_hash', sa.VARCHAR(length=60), autoincrement=False, nullable=False))
    # ### end Alembic commands ###
