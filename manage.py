from app.users.utils import load_model
from app import create_app

load_model()
app = create_app()
