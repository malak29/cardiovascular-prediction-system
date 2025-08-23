__version__ = "1.0.0"
__title__ = "Cardiovascular Disease Prediction System"
__description__ = "Production-ready ML system for predicting cardiovascular disease hospitalization rates"
__author__ = "Malak Parmar"
__email__ = "malakparmar.29@gmail.com"
__license__ = "MIT"

# Package-level imports for convenience
from app.core.config import get_settings
from app.core.database import get_db

__all__ = [
    "__version__",
    "__title__", 
    "__description__",
    "__author__",
    "__email__",
    "__license__",
    "get_settings",
    "get_db"
]