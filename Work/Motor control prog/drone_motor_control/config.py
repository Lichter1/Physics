"""
Configuration settings for Drone Motor Control application.
"""
import os


class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'drone-motor-control-dev-key')
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

    # MAVLink settings
    MAVLINK_CONNECTION = os.environ.get('MAVLINK_CONNECTION', 'tcp:127.0.0.1:8542')
    MAVLINK_BAUD = int(os.environ.get('MAVLINK_BAUD', '115220'))

    # PWM limits
    PWM_MIN = 1000
    PWM_MAX = 2000
    PWM_STOP = 1000

    # Motor configuration
    NUM_MOTORS = 4

    # Log locations
    SSD_LOG_PATH = os.environ.get('SSD_LOG_PATH', '/media/ssd/')
    COMMAND_LOG_PATH = 'data/command_logs/'
    CALIBRATION_PATH = 'data/calibration/'
    EXTRACTED_LOG_PATH = 'data/extracted_logs/'

    # Profile execution
    PROFILE_RESOLUTION_MS = 50  # Time resolution for profile generation
    DEFAULT_LOG_MARGIN_SEC = 5  # Seconds to add before/after log extraction
