"""
REST API endpoints for drone motor control.
"""
import io
import os
import threading
import zipfile
from datetime import datetime
from dateutil.parser import parse as parse_dt
from flask import Blueprint, jsonify, request, current_app, make_response

api_blueprint = Blueprint('api', __name__, url_prefix='/api')


@api_blueprint.errorhandler(Exception)
def handle_exception(e):
    """Return JSON for all errors."""
    import traceback
    print("[API ERROR] {}".format(e))
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500


def get_controller():
    """Get MAVLink controller from app context."""
    return current_app.config.get('mavlink_controller')


def get_profile_engine():
    """Get profile engine from app context."""
    return current_app.config.get('profile_engine')


def get_log_manager():
    """Get log manager from app context."""
    return current_app.config.get('log_manager')


def get_command_logger():
    """Get command logger from app context."""
    return current_app.config.get('command_logger')


def get_current_control():
    """Get current control module from app context."""
    return current_app.config.get('current_control')


def get_socketio():
    """Get SocketIO instance from app context."""
    return current_app.config.get('socketio')


# ============ Connection Management ============

@api_blueprint.route('/connection/status', methods=['GET'])
def get_connection_status():
    """Get MAVLink connection status."""
    print("[API] /connection/status called")
    controller = get_controller()
    if not controller:
        print("[API] Controller not initialized")
        return jsonify({'error': 'Controller not initialized'}), 500

    status = controller.get_status()
    print("[API] Status: connected={}".format(status.connected))
    return jsonify({
        'connected': status.connected,
        'system_id': status.system_id,
        'component_id': status.component_id,
        'last_heartbeat': status.last_heartbeat
    })


@api_blueprint.route('/connection/connect', methods=['POST'])
def connect():
    """Establish MAVLink connection."""
    print("[API] /connection/connect called")
    try:
        controller = get_controller()
        print("[API] Got controller: {}".format(controller))
        if not controller:
            print("[API] Controller not initialized")
            return jsonify({'error': 'Controller not initialized'}), 500

        if controller.is_connected:
            print("[API] Already connected")
            return jsonify({'message': 'Already connected', 'connected': True})

        # Safely get timeout
        timeout = 10.0
        try:
            if request.is_json and request.json:
                timeout = request.json.get('timeout', 10.0)
        except Exception as e:
            print("[API] Error getting timeout: {}".format(e))

        print("[API] Attempting connection with timeout={}".format(timeout))

        success = controller.connect(timeout=timeout)
        if success:
            print("[API] Connection successful")
            return jsonify({'message': 'Connected successfully', 'connected': True})
        else:
            print("[API] Connection failed - no heartbeat")
            return jsonify({'error': 'Connection failed - no heartbeat received', 'connected': False}), 400
    except Exception as e:
        import traceback
        print("[API] Connection error: {}".format(e))
        traceback.print_exc()
        return jsonify({'error': str(e), 'connected': False}), 500


@api_blueprint.route('/connection/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from MAVLink."""
    controller = get_controller()
    if not controller:
        return jsonify({'error': 'Controller not initialized'}), 500

    controller.disconnect()
    return jsonify({'message': 'Disconnected', 'connected': False})


# ============ Motor Control ============

@api_blueprint.route('/motors/stop', methods=['POST'])
def emergency_stop():
    """Emergency stop - set all motors to 1000."""
    controller = get_controller()
    if not controller:
        return jsonify({'error': 'Controller not initialized'}), 500

    # Abort any running profile
    engine = get_profile_engine()
    if engine and engine.is_running:
        engine.abort()

    try:
        controller.emergency_stop()
        return jsonify({'message': 'Emergency stop executed', 'pwm': 1000})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/motors/set', methods=['POST'])
def set_motor_pwm():
    """
    Set PWM for specified motors.
    Body: {"motor_ids": [1,2,3,4], "pwm": 1500}
    """
    controller = get_controller()
    if not controller:
        return jsonify({'error': 'Controller not initialized'}), 500

    if not controller.is_connected:
        return jsonify({'error': 'Not connected to MAVLink'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    motor_ids = data.get('motor_ids', [1, 2, 3, 4])
    pwm = data.get('pwm')

    if pwm is None:
        return jsonify({'error': 'PWM value required'}), 400

    try:
        pwm = int(pwm)
        if not 1000 <= pwm <= 2000:
            return jsonify({'error': 'PWM must be between 1000 and 2000'}), 400

        controller.set_selected_motors_pwm(motor_ids, pwm)
        return jsonify({
            'message': f'Set PWM {pwm} for motors {motor_ids}',
            'motor_ids': motor_ids,
            'pwm': pwm
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ Profile Management ============

@api_blueprint.route('/profiles', methods=['GET'])
def list_profiles():
    """List available motion profiles with parameter schemas."""
    engine = get_profile_engine()
    if not engine:
        return jsonify({'error': 'Profile engine not initialized'}), 500

    return jsonify(engine.list_profiles())


@api_blueprint.route('/profiles/preview', methods=['POST'])
def preview_profile():
    """
    Generate preview data for a profile.
    Body: {"profile": "Linear Ramp", "params": {...}}
    """
    engine = get_profile_engine()
    if not engine:
        return jsonify({'error': 'Profile engine not initialized'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    profile_name = data.get('profile')
    params = data.get('params', {})

    if not profile_name:
        return jsonify({'error': 'Profile name required'}), 400

    try:
        preview = engine.generate_preview(profile_name, params)
        return jsonify(preview)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@api_blueprint.route('/profiles/execute', methods=['POST'])
def execute_profile():
    """
    Execute a motion profile.
    Body: {
        "profile": "Linear Ramp",
        "params": {...},
        "motor_ids": [1,2,3,4]
    }
    """
    engine = get_profile_engine()
    controller = get_controller()
    command_logger = get_command_logger()
    socketio = get_socketio()

    if not engine:
        return jsonify({'error': 'Profile engine not initialized'}), 500
    if not controller:
        return jsonify({'error': 'Controller not initialized'}), 500
    if not controller.is_connected:
        return jsonify({'error': 'Not connected to MAVLink'}), 400

    if engine.is_running:
        return jsonify({'error': 'A profile is already executing'}), 400

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    profile_name = data.get('profile')
    params = data.get('params', {})
    motor_ids = data.get('motor_ids', [1, 2, 3, 4])

    if not profile_name:
        return jsonify({'error': 'Profile name required'}), 400

    def progress_callback(state):
        """Send progress updates via WebSocket."""
        if socketio:
            socketio.emit('execution_progress', {
                'running': state.running,
                'elapsed_sec': state.elapsed_sec,
                'total_sec': state.total_sec,
                'current_pwm': state.current_pwm,
                'progress_pct': state.progress_pct,
                'session_id': state.session_id,
                'current_iteration': state.current_iteration,
                'total_iterations': state.total_iterations
            })

            # Emit loop progress if looping
            if state.current_iteration is not None and state.total_iterations is not None:
                socketio.emit('loop_progress', {
                    'iteration': state.current_iteration,
                    'total_iterations': state.total_iterations,
                    'session_id': state.session_id
                })

            # Emit error notification if motor stop failed
            if state.error and state.failed_motors:
                socketio.emit('motor_stop_failed', {
                    'failed_motors': state.failed_motors,
                    'message': state.error,
                    'session_id': state.session_id
                })

    def run_execution():
        """Run profile execution in background thread."""
        try:
            session_id = engine.execute_sync(
                profile_name, params, motor_ids,
                controller, command_logger,
                progress_callback
            )
            if socketio:
                socketio.emit('execution_complete', {
                    'session_id': session_id,
                    'message': 'Profile execution completed'
                })
        except Exception as e:
            if socketio:
                socketio.emit('execution_error', {'error': str(e)})

    # Start execution in background thread
    thread = threading.Thread(target=run_execution)
    thread.daemon = True
    thread.start()

    return jsonify({
        'message': 'Profile execution started',
        'profile': profile_name,
        'motor_ids': motor_ids,
        'status': 'started'
    })


@api_blueprint.route('/profiles/abort', methods=['POST'])
def abort_execution():
    """Abort currently running profile."""
    engine = get_profile_engine()
    if not engine:
        return jsonify({'error': 'Profile engine not initialized'}), 500

    if not engine.is_running:
        return jsonify({'message': 'No profile is running'})

    engine.abort()
    return jsonify({'message': 'Abort requested'})


# ============ Log Management ============

@api_blueprint.route('/logs/discover', methods=['GET'])
def discover_logs():
    """Find latest log files on SSD."""
    log_manager = get_log_manager()
    if not log_manager:
        return jsonify({'error': 'Log manager not initialized'}), 500

    latest = log_manager.discover_latest_logs()

    result = {}
    for log_type, log_file in latest.items():
        if log_file:
            result[log_type] = {
                'path': log_file.path,
                'timestamp': log_file.timestamp.isoformat(),
                'size_bytes': log_file.size_bytes
            }
        else:
            result[log_type] = None

    return jsonify(result)


@api_blueprint.route('/logs/extract', methods=['POST'])
def extract_logs():
    """
    Extract time-sliced portions from logs.
    Body: {
        "start_time": "ISO8601",
        "end_time": "ISO8601",
        "margin_seconds": 5,
        "log_types": ["esc_telemetry", "temps_server"]
    }
    """
    log_manager = get_log_manager()
    if not log_manager:
        return jsonify({'error': 'Log manager not initialized'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    try:
        # parse_dt handles Z suffix and timezone offsets that fromisoformat rejects on Python<3.11
        start_time = parse_dt(data['start_time']).replace(tzinfo=None)
        end_time = parse_dt(data['end_time']).replace(tzinfo=None)
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid time format: {e}'}), 400

    margin_seconds = data.get('margin_seconds', 5)
    log_types = data.get('log_types')  # None means all

    # Get output directory from config
    output_dir = current_app.config.get('EXTRACTED_LOG_PATH', 'data/extracted_logs/')

    try:
        results = log_manager.extract_all_logs(
            start_time, end_time, output_dir,
            margin_seconds, log_types
        )

        extracted = []
        for log_type, path in results.items():
            if path:
                extracted.append({
                    'type': log_type,
                    'path': path,
                    'size_bytes': os.path.getsize(path) if os.path.exists(path) else 0
                })

        return jsonify({
            'extracted_files': extracted,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/logs/command-history', methods=['GET'])
def get_command_logs():
    """List available command log sessions."""
    command_logger = get_command_logger()
    if not command_logger:
        return jsonify({'error': 'Command logger not initialized'}), 500

    sessions = command_logger.list_sessions()
    return jsonify(sessions)


@api_blueprint.route('/logs/command-history/<session_id>', methods=['GET'])
def get_command_session(session_id):
    """Get details of a specific command log session."""
    command_logger = get_command_logger()
    if not command_logger:
        return jsonify({'error': 'Command logger not initialized'}), 500

    session = command_logger.load_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify(session)


@api_blueprint.route('/logs/export-session/<session_id>', methods=['GET'])
def export_session_logs(session_id):
    """
    Export log files for a command session as a zip download.

    Query params:
        margin_seconds: seconds to include before/after session (default 5)
        log_types: comma-separated list of types to include (default all)
                   e.g. esc_telemetry,temps_server,pdb_server,general
    """
    command_logger = get_command_logger()
    log_manager = get_log_manager()

    if not command_logger:
        return jsonify({'error': 'Command logger not initialized'}), 500
    if not log_manager:
        return jsonify({'error': 'Log manager not initialized'}), 500

    # Load session data
    session = command_logger.load_session(session_id)
    if not session:
        return jsonify({'error': f'Session not found: {session_id}'}), 404

    start_data = session.get('start')
    end_data = session.get('end')

    if not start_data:
        return jsonify({'error': 'Session has no start record'}), 400

    start_time = datetime.fromisoformat(start_data['timestamp'])

    if end_data:
        end_time = datetime.fromisoformat(end_data['timestamp'])
    elif session.get('commands'):
        end_time = datetime.fromisoformat(session['commands'][-1]['timestamp'])
    else:
        end_time = start_time

    # Parse query parameters
    try:
        margin_seconds = float(request.args.get('margin_seconds', 5))
    except ValueError:
        margin_seconds = 5.0

    log_types_param = request.args.get('log_types', '')
    requested_log_types = [lt.strip() for lt in log_types_param.split(',') if lt.strip()]
    if not requested_log_types:
        requested_log_types = list(log_manager.PATTERNS.keys())

    # Build zip in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Include the command log file itself
        cmd_log_filename = f"{session_id}_commands.jsonl"
        cmd_log_path = os.path.join(command_logger.output_dir, cmd_log_filename)
        if os.path.exists(cmd_log_path):
            zf.write(cmd_log_path, cmd_log_filename)

        # Find and extract SSD log files for the session time window
        all_logs_by_type = log_manager.discover_all_logs()

        for log_type in requested_log_types:
            logs_of_type = all_logs_by_type.get(log_type, [])

            # Find the log file that was active during the session:
            # pick the most recent file whose start timestamp is <= session end
            candidate = None
            for log_file in sorted(logs_of_type, key=lambda x: x.timestamp):
                if log_file.timestamp <= end_time:
                    candidate = log_file

            if candidate:
                content = log_manager.extract_time_slice(
                    candidate, start_time, end_time, margin_seconds
                )
                if content:
                    extracted_filename = f"{session_id}_extracted_{log_type}.log"
                    zf.writestr(extracted_filename, content)

    zip_buffer.seek(0)
    zip_data = zip_buffer.read()
    zip_filename = f"session_{session_id}_logs.zip"

    # Use make_response with raw bytes instead of send_file(BytesIO) to avoid
    # eventlet monkey-patching incompatibilities with streaming responses.
    response = make_response(zip_data)
    response.headers['Content-Type'] = 'application/zip'
    response.headers['Content-Disposition'] = f'attachment; filename="{zip_filename}"'
    response.headers['Content-Length'] = len(zip_data)
    return response


# ============ Current Control ============

@api_blueprint.route('/current-control/calibrate', methods=['POST'])
def start_calibration():
    """
    Start a calibration run.
    Body: {
        "motor_ids": [1],
        "pwm_min": 1000,
        "pwm_max": 1800,
        "rate": 20
    }
    """
    current_control = get_current_control()
    engine = get_profile_engine()

    if not current_control:
        return jsonify({'error': 'Current control not initialized'}), 500
    if not engine:
        return jsonify({'error': 'Profile engine not initialized'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    pwm_min = data.get('pwm_min', 1000)
    pwm_max = data.get('pwm_max', 1800)
    rate = data.get('rate', 20)

    # Generate calibration profile
    profile_config = current_control.generate_calibration_profile(pwm_min, pwm_max, rate)

    return jsonify({
        'message': 'Use this profile configuration to run calibration',
        'profile_config': profile_config,
        'instructions': 'Execute this profile, then call /current-control/process with the session_id'
    })


@api_blueprint.route('/current-control/process', methods=['POST'])
def process_calibration():
    """
    Process completed calibration run.
    Body: {
        "session_id": str,
        "motor_id": int
    }
    """
    current_control = get_current_control()
    command_logger = get_command_logger()
    log_manager = get_log_manager()

    if not current_control:
        return jsonify({'error': 'Current control not initialized'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    session_id = data.get('session_id')
    motor_id = data.get('motor_id')

    if not session_id or not motor_id:
        return jsonify({'error': 'session_id and motor_id are required'}), 400

    # Load session to get time range
    session = command_logger.load_session(session_id)
    if not session or not session.get('start') or not session.get('end'):
        return jsonify({'error': 'Session not found or incomplete'}), 404

    start_time = datetime.fromisoformat(session['start']['timestamp'])
    end_time = datetime.fromisoformat(session['end']['timestamp'])

    # Find ESC telemetry log
    logs = log_manager.discover_logs_near_time(start_time)
    esc_log = logs.get('esc_telemetry')

    if not esc_log:
        return jsonify({'error': 'ESC telemetry log not found'}), 404

    # Parse calibration data
    try:
        curve = current_control.parse_calibration_run(
            esc_log.path, start_time, end_time, motor_id
        )

        if not curve:
            return jsonify({'error': 'No calibration data found in log'}), 400

        # Fit and save
        current_control.fit_curve(curve)
        filepath = current_control.save_calibration(curve)

        plot_data = current_control.get_calibration_plot_data(motor_id)

        return jsonify({
            'message': 'Calibration processed successfully',
            'motor_id': motor_id,
            'num_points': len(curve.points),
            'pwm_range': [curve.pwm_min, curve.pwm_max],
            'saved_to': filepath,
            'plot_data': plot_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_blueprint.route('/current-control/calibrations', methods=['GET'])
def list_calibrations():
    """List available calibration curves per motor."""
    current_control = get_current_control()
    if not current_control:
        return jsonify({'error': 'Current control not initialized'}), 500

    return jsonify(current_control.list_calibrations())


@api_blueprint.route('/current-control/calibrations/<int:motor_id>', methods=['GET'])
def get_calibration(motor_id):
    """Get calibration curve data for a motor."""
    current_control = get_current_control()
    if not current_control:
        return jsonify({'error': 'Current control not initialized'}), 500

    plot_data = current_control.get_calibration_plot_data(motor_id)
    if not plot_data:
        return jsonify({'error': f'No calibration found for motor {motor_id}'}), 404

    return jsonify(plot_data)


@api_blueprint.route('/current-control/lookup', methods=['POST'])
def current_to_pwm_lookup():
    """
    Look up PWM for target current.
    Body: {"motor_id": int, "target_current": float}
    """
    current_control = get_current_control()
    if not current_control:
        return jsonify({'error': 'Current control not initialized'}), 500

    data = request.json
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    motor_id = data.get('motor_id')
    target_current = data.get('target_current')

    if motor_id is None or target_current is None:
        return jsonify({'error': 'motor_id and target_current are required'}), 400

    if not current_control.has_calibration(motor_id):
        return jsonify({'error': f'No calibration found for motor {motor_id}'}), 404

    pwm = current_control.current_to_pwm(motor_id, float(target_current))
    if pwm is None:
        return jsonify({'error': 'Target current is outside calibrated range'}), 400

    # Also return expected current for verification
    expected_current = current_control.pwm_to_current(motor_id, pwm)

    return jsonify({
        'motor_id': motor_id,
        'target_current': target_current,
        'pwm': pwm,
        'expected_current': expected_current
    })
