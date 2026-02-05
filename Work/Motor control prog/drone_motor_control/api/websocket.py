"""
WebSocket handlers for real-time updates.
"""
from flask_socketio import SocketIO, emit


def init_websocket(socketio: SocketIO, app):
    """
    Initialize WebSocket event handlers.

    Args:
        socketio: Flask-SocketIO instance
        app: Flask application
    """

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print('Client connected')
        # Send current connection status
        controller = app.config.get('mavlink_controller')
        if controller:
            status = controller.get_status()
            emit('connection_status', {
                'connected': status.connected,
                'system_id': status.system_id,
                'component_id': status.component_id
            })

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print('Client disconnected')

    @socketio.on('subscribe_telemetry')
    def handle_subscribe_telemetry():
        """
        Handle request to start receiving telemetry updates.
        Note: Real-time ESC telemetry would require additional implementation
        to read from ESC logs or a live telemetry stream.
        """
        emit('telemetry_status', {'subscribed': True, 'message': 'Subscribed to telemetry'})

    @socketio.on('unsubscribe_telemetry')
    def handle_unsubscribe_telemetry():
        """Handle request to stop receiving telemetry updates."""
        emit('telemetry_status', {'subscribed': False, 'message': 'Unsubscribed from telemetry'})

    @socketio.on('request_status')
    def handle_request_status():
        """Handle request for current system status."""
        controller = app.config.get('mavlink_controller')
        engine = app.config.get('profile_engine')

        status = {
            'connected': False,
            'executing': False
        }

        if controller:
            conn_status = controller.get_status()
            status['connected'] = conn_status.connected
            status['system_id'] = conn_status.system_id
            status['component_id'] = conn_status.component_id

        if engine:
            status['executing'] = engine.is_running

        emit('system_status', status)

    return socketio
