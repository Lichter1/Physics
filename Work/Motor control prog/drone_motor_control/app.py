"""
Main Flask application for Drone Motor Control.
"""
# CRITICAL: Monkey patch MUST happen before any other imports
# This allows eventlet to work with blocking I/O (pymavlink sockets)
import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template
from flask_socketio import SocketIO

from config import Config
from core import (
    MAVLinkController,
    ProfileEngine,
    LogManager,
    CommandLogger,
    CurrentControlModule
)
from api import api_blueprint, init_websocket


def create_app(config_class=Config):
    """
    Application factory.

    Args:
        config_class: Configuration class to use

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

    # Initialize core components
    mavlink_controller = MAVLinkController(
        connection_string=app.config['MAVLINK_CONNECTION'],
        baud=app.config['MAVLINK_BAUD']
    )

    profile_engine = ProfileEngine()

    log_manager = LogManager(
        ssd_path=app.config['SSD_LOG_PATH']
    )

    command_logger = CommandLogger(
        output_dir=app.config['COMMAND_LOG_PATH']
    )

    current_control = CurrentControlModule(
        calibration_dir=app.config['CALIBRATION_PATH'],
        log_manager=log_manager
    )

    # Store in app config for access in routes
    app.config['mavlink_controller'] = mavlink_controller
    app.config['profile_engine'] = profile_engine
    app.config['log_manager'] = log_manager
    app.config['command_logger'] = command_logger
    app.config['current_control'] = current_control
    app.config['socketio'] = socketio
    app.config['EXTRACTED_LOG_PATH'] = Config.EXTRACTED_LOG_PATH

    # Register blueprints
    app.register_blueprint(api_blueprint)

    # Initialize WebSocket handlers
    init_websocket(socketio, app)

    # Main route
    @app.route('/')
    def index():
        return render_template('index.html')

    return app, socketio


def main():
    """Run the application."""
    app, socketio = create_app()

    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')

    print("")
    print("========================================")
    print("Drone Motor Control Web Interface")
    print("========================================")
    print("")
    print("Server starting on http://{}:{}".format(host, port))
    print("")
    print("MAVLink connection: {}".format(app.config['MAVLINK_CONNECTION']))
    print("  (Set MAVLINK_CONNECTION env var to change)")
    print("")
    print("To access from another machine via SSH tunnel:")
    print("ssh -L {}:localhost:{} user@drone-ip".format(port, port))
    print("Then open: http://localhost:{}".format(port))
    print("")
    print("========================================")

    socketio.run(app, host=host, port=port, debug=Config.DEBUG)


if __name__ == '__main__':
    main()
