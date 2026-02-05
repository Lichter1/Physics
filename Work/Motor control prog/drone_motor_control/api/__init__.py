"""
API module for drone motor control.
"""
from .routes import api_blueprint
from .websocket import init_websocket

__all__ = ['api_blueprint', 'init_websocket']
