/**
 * Drone Motor Control - Main JavaScript Application
 */

const DroneControl = {
    socket: null,
    previewChart: null,
    currentChart: null,
    profiles: [],
    isExecuting: false,

    /**
     * Initialize the application
     */
    init() {
        this.initSocket();
        this.initEventListeners();
        this.loadProfiles();
        this.initPreviewChart();
        this.initCurrentChart();
        this.checkConnectionStatus();
        // Initialize SequenceBuilder after DOM is ready
        SequenceBuilder.init(this);
    },

    /**
     * Initialize WebSocket connection
     */
    initSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('WebSocket connected');
        });

        this.socket.on('disconnect', () => {
            console.log('WebSocket disconnected');
        });

        this.socket.on('connection_status', (data) => {
            this.updateConnectionUI(data.connected);
        });

        this.socket.on('execution_progress', (data) => {
            this.handleProgress(data);
        });

        this.socket.on('execution_complete', (data) => {
            this.handleComplete(data);
        });

        this.socket.on('execution_error', (data) => {
            this.handleError(data);
        });

        this.socket.on('motor_stop_failed', (data) => {
            this.handleMotorStopFailure(data);
        });

        this.socket.on('loop_progress', (data) => {
            this.handleLoopProgress(data);
        });
    },

    /**
     * Initialize event listeners
     */
    initEventListeners() {
        // Tab navigation
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // Connection
        document.getElementById('connect-btn').addEventListener('click', () => this.toggleConnection());

        // Emergency stop
        document.getElementById('emergency-stop').addEventListener('click', () => this.emergencyStop());

        // Motor selection
        document.getElementById('select-all').addEventListener('click', () => this.selectAllMotors(true));
        document.getElementById('select-none').addEventListener('click', () => this.selectAllMotors(false));

        // Profile
        document.getElementById('profile-select').addEventListener('change', (e) => this.onProfileChange(e.target.value));
        document.getElementById('generate-preview').addEventListener('click', () => this.generatePreview());
        document.getElementById('execute-btn').addEventListener('click', () => this.executeProfile());
        document.getElementById('abort-btn').addEventListener('click', () => this.abortExecution());

        // Logs
        document.getElementById('discover-logs').addEventListener('click', () => this.discoverLogs());
        document.getElementById('extract-logs').addEventListener('click', () => this.extractLogs());
        document.getElementById('load-command-history').addEventListener('click', () => this.loadCommandHistory());

        // Current control
        document.getElementById('run-calibration').addEventListener('click', () => this.runCalibration());
        document.getElementById('process-calibration').addEventListener('click', () => this.processCalibration());
        document.getElementById('load-curve').addEventListener('click', () => this.loadCalibrationCurve());
        document.getElementById('lookup-pwm').addEventListener('click', () => this.lookupPWM());
    },

    /**
     * Initialize preview chart
     */
    initPreviewChart() {
        const ctx = document.getElementById('preview-chart').getContext('2d');
        this.previewChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PWM',
                    data: [],
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Time (sec)', color: '#888' },
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    },
                    y: {
                        title: { display: true, text: 'PWM', color: '#888' },
                        min: 1000,
                        max: 2000,
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    },

    /**
     * Initialize current chart
     */
    initCurrentChart() {
        const ctx = document.getElementById('current-chart').getContext('2d');
        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Current (A)',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'PWM', color: '#888' },
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    },
                    y: {
                        title: { display: true, text: 'Current (A)', color: '#888' },
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    },

    /**
     * Switch between tabs
     */
    switchTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabId}-tab`);
        });
    },

    /**
     * Check connection status
     */
    async checkConnectionStatus() {
        try {
            const response = await fetch('/api/connection/status');
            const data = await response.json();
            this.updateConnectionUI(data.connected);
        } catch (error) {
            console.error('Error checking connection:', error);
            this.updateConnectionUI(false);
        }
    },

    /**
     * Update connection UI
     */
    updateConnectionUI(connected) {
        const indicator = document.getElementById('conn-indicator');
        const text = document.getElementById('conn-text');
        const btn = document.getElementById('connect-btn');

        indicator.className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
        text.textContent = connected ? 'Connected' : 'Disconnected';
        btn.textContent = connected ? 'Disconnect' : 'Connect';
    },

    /**
     * Toggle MAVLink connection
     */
    async toggleConnection() {
        const btn = document.getElementById('connect-btn');
        const isConnected = btn.textContent === 'Disconnect';

        btn.disabled = true;
        btn.textContent = isConnected ? 'Disconnecting...' : 'Connecting...';

        try {
            const endpoint = isConnected ? '/api/connection/disconnect' : '/api/connection/connect';
            const response = await fetch(endpoint, { method: 'POST' });
            const data = await response.json();
            this.updateConnectionUI(data.connected);
        } catch (error) {
            console.error('Connection error:', error);
            alert('Connection error: ' + error.message);
        } finally {
            btn.disabled = false;
        }
    },

    /**
     * Emergency stop
     */
    async emergencyStop() {
        try {
            const response = await fetch('/api/motors/stop', { method: 'POST' });
            const data = await response.json();
            alert(data.message || 'Emergency stop executed');
        } catch (error) {
            console.error('Emergency stop error:', error);
            alert('Emergency stop error: ' + error.message);
        }
    },

    /**
     * Select/deselect all motors
     */
    selectAllMotors(select) {
        document.querySelectorAll('input[name="motor"]').forEach(cb => {
            cb.checked = select;
        });
    },

    /**
     * Get selected motor IDs
     */
    getSelectedMotors() {
        const selected = [];
        document.querySelectorAll('input[name="motor"]:checked').forEach(cb => {
            selected.push(parseInt(cb.value));
        });
        return selected;
    },

    /**
     * Load available profiles
     */
    async loadProfiles() {
        try {
            const response = await fetch('/api/profiles');
            this.profiles = await response.json();

            const select = document.getElementById('profile-select');
            select.innerHTML = this.profiles.map(p =>
                `<option value="${p.name}">${p.name}</option>`
            ).join('');

            if (this.profiles.length > 0) {
                this.onProfileChange(this.profiles[0].name);
            }
        } catch (error) {
            console.error('Error loading profiles:', error);
        }
    },

    /**
     * Handle profile selection change
     */
    onProfileChange(profileName) {
        const profile = this.profiles.find(p => p.name === profileName);
        if (!profile) return;

        const isMultiSeq = (profileName === "Multi-Sequence");

        // Show/hide appropriate UI
        const paramsContainer = document.getElementById('profile-params');
        const multiSeqBuilder = document.getElementById('multi-sequence-builder');

        if (isMultiSeq) {
            // Show multi-sequence builder, hide single profile params
            paramsContainer.style.display = 'none';
            multiSeqBuilder.style.display = 'block';

            // Initialize with one sequence if empty
            if (SequenceBuilder.sequences.length === 0) {
                SequenceBuilder.addSequence();
            } else {
                SequenceBuilder.render();
            }
        } else {
            // Show single profile params, hide multi-sequence builder
            paramsContainer.style.display = 'block';
            multiSeqBuilder.style.display = 'none';

            // Generate parameter inputs for single profile
            paramsContainer.innerHTML = '';
            for (const [key, param] of Object.entries(profile.parameters)) {
                if (param.type === 'array') {
                    // Handle array type (custom steps)
                    paramsContainer.innerHTML += this.createStepsInput(key, param);
                } else {
                    paramsContainer.innerHTML += this.createParamInput(key, param);
                }
            }
        }

        // Reset preview
        document.getElementById('execute-btn').disabled = true;
    },

    /**
     * Create parameter input HTML
     */
    createParamInput(key, param) {
        const id = `param-${key}`;
        let input = '';

        if (param.type === 'bool') {
            input = `<input type="checkbox" id="${id}" ${param.default ? 'checked' : ''}>`;
        } else {
            const inputType = param.type === 'int' ? 'number' : 'number';
            const step = param.type === 'float' ? '0.1' : '1';
            input = `<input type="${inputType}" id="${id}" class="form-control"
                     value="${param.default || ''}"
                     min="${param.min || ''}" max="${param.max || ''}" step="${step}">`;
        }

        return `
            <div class="form-group">
                <label for="${id}">${param.label || key}</label>
                ${input}
            </div>
        `;
    },

    /**
     * Create steps array input for custom steps profile
     */
    createStepsInput(key, param) {
        return `
            <div class="form-group">
                <label>Steps</label>
                <div id="steps-container">
                    <div class="step-row">
                        <input type="number" class="form-control step-pwm" placeholder="PWM" value="1500" min="1000" max="2000">
                        <input type="number" class="form-control step-duration" placeholder="Duration (sec)" value="5" min="0.1" max="60" step="0.1">
                        <button type="button" class="btn btn-danger btn-small remove-step">X</button>
                    </div>
                </div>
                <button type="button" id="add-step" class="btn btn-secondary btn-small">Add Step</button>
            </div>
            <div class="form-group">
                <label><input type="checkbox" id="param-loop"> Loop Forever</label>
            </div>
        `;
    },

    /**
     * Collect profile parameters from form
     */
    collectParams() {
        const profileName = document.getElementById('profile-select').value;

        // Handle multi-sequence profiles
        if (profileName === "Multi-Sequence") {
            return SequenceBuilder.collectParams();
        }

        // Handle single profiles
        const profile = this.profiles.find(p => p.name === profileName);
        if (!profile) return {};

        const params = {};

        for (const [key, param] of Object.entries(profile.parameters)) {
            if (param.type === 'array') {
                // Collect steps
                const steps = [];
                document.querySelectorAll('.step-row').forEach(row => {
                    const pwm = parseInt(row.querySelector('.step-pwm').value);
                    const duration = parseFloat(row.querySelector('.step-duration').value);
                    if (!isNaN(pwm) && !isNaN(duration)) {
                        steps.push({ pwm, duration_sec: duration });
                    }
                });
                params[key] = steps;
            } else if (param.type === 'bool') {
                params[key] = document.getElementById(`param-${key}`).checked;
            } else if (param.type === 'int') {
                params[key] = parseInt(document.getElementById(`param-${key}`).value);
            } else {
                params[key] = parseFloat(document.getElementById(`param-${key}`).value);
            }
        }

        return params;
    },

    /**
     * Generate profile preview
     */
    async generatePreview() {
        const profileName = document.getElementById('profile-select').value;
        const params = this.collectParams();

        // Warn for infinite loop
        if (params.loop_count === -1) {
            if (!confirm('⚠️ This will create an INFINITE loop.\n\nYou must manually abort to stop execution.\n\nContinue with preview?')) {
                return;
            }
        }

        try {
            const response = await fetch('/api/profiles/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile: profileName, params })
            });

            if (!response.ok) {
                const error = await response.json();
                alert('Error: ' + error.error);
                return;
            }

            const data = await response.json();
            this.updatePreviewChart(data);
            document.getElementById('execute-btn').disabled = false;
        } catch (error) {
            console.error('Preview error:', error);
            alert('Error generating preview: ' + error.message);
        }
    },

    /**
     * Update preview chart with data
     */
    updatePreviewChart(data) {
        // Sample data points for better performance
        const maxPoints = 200;
        let chartData = data.data;
        if (chartData.length > maxPoints) {
            const step = Math.ceil(chartData.length / maxPoints);
            chartData = chartData.filter((_, i) => i % step === 0);
        }

        this.previewChart.data.labels = chartData.map(p => p.t.toFixed(2));
        this.previewChart.data.datasets[0].data = chartData.map(p => p.pwm);

        // Add sequence boundary annotations if available (multi-sequence profiles)
        if (data.sequence_boundaries && data.sequence_labels) {
            const annotations = {};
            data.sequence_boundaries.forEach((time, i) => {
                if (i > 0) {  // Skip first boundary (t=0)
                    annotations[`boundary${i}`] = {
                        type: 'line',
                        xMin: time,
                        xMax: time,
                        borderColor: 'rgba(255, 99, 132, 0.6)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        label: {
                            content: data.sequence_labels[i] || `Seq ${i}`,
                            enabled: true,
                            position: 'start',
                            backgroundColor: 'rgba(255, 99, 132, 0.8)',
                            color: 'white',
                            font: {
                                size: 10
                            }
                        }
                    };
                }
            });

            // Update chart options with annotations
            if (!this.previewChart.options.plugins) {
                this.previewChart.options.plugins = {};
            }
            this.previewChart.options.plugins.annotation = { annotations };
        } else {
            // Clear annotations if not multi-sequence
            if (this.previewChart.options.plugins && this.previewChart.options.plugins.annotation) {
                this.previewChart.options.plugins.annotation = { annotations: {} };
            }
        }

        this.previewChart.update();

        // Update duration and point count info
        let durationText = data.duration.toFixed(1);
        const loopCount = document.getElementById('loop-count')?.value;
        if (loopCount && parseInt(loopCount) !== 0) {
            if (parseInt(loopCount) === -1) {
                durationText += ' (per iteration, ∞ loop)';
            } else {
                const totalDuration = data.duration * parseInt(loopCount);
                durationText += ` (per iteration, ${totalDuration.toFixed(1)}s total)`;
            }
        }
        document.getElementById('duration-value').textContent = durationText;
        document.getElementById('points-value').textContent = data.data.length;
    },

    /**
     * Execute profile
     */
    async executeProfile() {
        const profileName = document.getElementById('profile-select').value;
        const params = this.collectParams();
        const motorIds = this.getSelectedMotors();

        if (motorIds.length === 0) {
            alert('Please select at least one motor');
            return;
        }

        if (!confirm(`Execute ${profileName} on motors ${motorIds.join(', ')}?`)) {
            return;
        }

        try {
            const response = await fetch('/api/profiles/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile: profileName, params, motor_ids: motorIds })
            });

            const data = await response.json();
            if (response.ok) {
                this.isExecuting = true;
                document.getElementById('execution-panel').style.display = 'block';
                document.getElementById('execute-btn').disabled = true;
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Execution error:', error);
            alert('Error starting execution: ' + error.message);
        }
    },

    /**
     * Handle execution progress
     */
    handleProgress(data) {
        document.getElementById('progress-fill').style.width = `${data.progress_pct}%`;
        document.getElementById('progress-text').textContent = `${Math.round(data.progress_pct)}%`;
        document.getElementById('elapsed-value').textContent = data.elapsed_sec.toFixed(1);
        document.getElementById('current-pwm-value').textContent = data.current_pwm;

        // Show/hide loop info based on whether we're looping
        const loopInfoContainer = document.getElementById('loop-info-container');
        if (data.current_iteration !== null && data.total_iterations !== null) {
            loopInfoContainer.style.display = 'inline';
            const loopInfo = data.total_iterations === -1
                ? `Iteration ${data.current_iteration + 1}`
                : `Iteration ${data.current_iteration + 1}/${data.total_iterations}`;
            document.getElementById('loop-info').textContent = loopInfo;
        } else {
            loopInfoContainer.style.display = 'none';
        }
    },

    /**
     * Handle loop progress updates
     */
    handleLoopProgress(data) {
        console.log('Loop progress:', data);
        // Loop info is now handled in handleProgress
    },

    /**
     * Handle execution complete
     */
    handleComplete(data) {
        this.isExecuting = false;
        document.getElementById('execution-panel').style.display = 'none';
        document.getElementById('execute-btn').disabled = false;
        alert('Execution complete! Session ID: ' + data.session_id);
    },

    /**
     * Handle execution error
     */
    handleError(data) {
        this.isExecuting = false;
        document.getElementById('execution-panel').style.display = 'none';
        document.getElementById('execute-btn').disabled = false;
        alert('Execution error: ' + data.error);
    },

    /**
     * Handle motor stop failure (critical safety issue)
     */
    handleMotorStopFailure(data) {
        // Display persistent critical error banner
        const banner = document.createElement('div');
        banner.id = 'motor-stop-error-banner';
        banner.className = 'error-banner';
        banner.innerHTML = `
            <div class="error-content">
                <strong>⚠️ CRITICAL: Motor Stop Failed!</strong>
                <p>Motors ${data.failed_motors.join(', ')} failed to stop. ${data.message}</p>
                <p>Check MAVLink connection and manually verify motors are stopped!</p>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;

        // Insert at top of page
        const container = document.querySelector('.container');
        container.insertBefore(banner, container.firstChild);

        // Disable execute button until dismissed
        document.getElementById('execute-btn').disabled = true;

        // Log to console
        console.error('[CRITICAL] Motor stop failed:', data);

        // Also show alert for immediate attention
        alert(`⚠️ CRITICAL: Motors ${data.failed_motors.join(', ')} failed to stop!\n\n${data.message}\n\nCheck MAVLink connection and manually verify motors are stopped!`);
    },

    /**
     * Abort execution
     */
    async abortExecution() {
        try {
            await fetch('/api/profiles/abort', { method: 'POST' });
        } catch (error) {
            console.error('Abort error:', error);
        }
    },

    /**
     * Discover log files
     */
    async discoverLogs() {
        try {
            const response = await fetch('/api/logs/discover');
            const data = await response.json();

            const container = document.getElementById('discovered-logs');
            let html = '';

            for (const [type, info] of Object.entries(data)) {
                if (info) {
                    const date = new Date(info.timestamp).toLocaleString();
                    const size = (info.size_bytes / 1024).toFixed(1);
                    html += `
                        <div class="log-item">
                            <div class="log-type">${type}</div>
                            <div class="log-time">${date}</div>
                            <div class="log-path">${info.path} (${size} KB)</div>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="log-item">
                            <div class="log-type">${type}</div>
                            <div class="log-path">Not found</div>
                        </div>
                    `;
                }
            }

            container.innerHTML = html || '<p class="placeholder">No logs found</p>';
        } catch (error) {
            console.error('Error discovering logs:', error);
            alert('Error discovering logs: ' + error.message);
        }
    },

    /**
     * Extract logs
     */
    async extractLogs() {
        const startTime = document.getElementById('extract-start').value;
        const endTime = document.getElementById('extract-end').value;
        const margin = parseInt(document.getElementById('extract-margin').value);

        if (!startTime || !endTime) {
            alert('Please specify start and end times');
            return;
        }

        const logTypes = [];
        document.querySelectorAll('input[name="log-type"]:checked').forEach(cb => {
            logTypes.push(cb.value);
        });

        try {
            const response = await fetch('/api/logs/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    start_time: new Date(startTime).toISOString(),
                    end_time: new Date(endTime).toISOString(),
                    margin_seconds: margin,
                    log_types: logTypes
                })
            });

            const data = await response.json();
            const container = document.getElementById('extraction-results');

            if (response.ok) {
                container.className = 'result-success';
                container.innerHTML = `
                    <strong>Extracted ${data.extracted_files.length} files:</strong>
                    <ul>
                        ${data.extracted_files.map(f => `<li>${f.type}: ${f.path}</li>`).join('')}
                    </ul>
                `;
            } else {
                container.className = 'result-error';
                container.innerHTML = `<strong>Error:</strong> ${data.error}`;
            }
        } catch (error) {
            console.error('Extraction error:', error);
            document.getElementById('extraction-results').innerHTML = `<strong>Error:</strong> ${error.message}`;
        }
    },

    /**
     * Load command history
     */
    async loadCommandHistory() {
        try {
            const response = await fetch('/api/logs/command-history');
            const sessions = await response.json();

            const container = document.getElementById('command-sessions');

            if (!Array.isArray(sessions) || sessions.length === 0) {
                container.innerHTML = '<p class="placeholder">No command sessions found</p>';
                return;
            }

            container.innerHTML = sessions.map(s => {
                const startStr = new Date(s.timestamp).toLocaleString();
                const endStr = s.end_time ? new Date(s.end_time).toLocaleString() : '—';
                const dur = s.duration_sec != null ? `${s.duration_sec.toFixed(1)} s` : '—';
                const cmds = s.total_commands != null ? `${s.total_commands} cmds` : '';
                return `
                <div class="session-item">
                    <div class="session-info">
                        <div class="session-profile"><strong>${s.profile_name}</strong> &mdash; Motors: ${s.motor_ids.join(', ')}</div>
                        <div class="session-time">${startStr} &rarr; ${endStr}</div>
                        <div class="session-meta">${dur}${cmds ? ' &bull; ' + cmds : ''}</div>
                    </div>
                    <button class="btn btn-primary btn-small session-export-btn"
                            onclick="app.exportSessionLogs('${s.session_id}')">
                        &#8681; Export Logs
                    </button>
                </div>`;
            }).join('');
        } catch (error) {
            console.error('Error loading command history:', error);
        }
    },

    /**
     * Export SSD logs for a command session as a zip download.
     */
    async exportSessionLogs(sessionId) {
        const margin = parseInt(document.getElementById('history-margin').value) || 5;
        const logTypes = [];
        document.querySelectorAll('input[name="history-log-type"]:checked').forEach(cb => {
            logTypes.push(cb.value);
        });

        const params = new URLSearchParams({
            margin_seconds: margin,
            log_types: logTypes.join(',')
        });

        const url = `/api/logs/export-session/${sessionId}?${params.toString()}`;

        // Find the clicked button to show progress feedback
        const btn = document.querySelector(`.session-export-btn[onclick*="${sessionId}"]`);
        const origText = btn ? btn.innerHTML : '';
        if (btn) { btn.textContent = 'Downloading\u2026'; btn.disabled = true; }

        try {
            const response = await fetch(url);

            if (!response.ok) {
                let errMsg = response.statusText;
                try {
                    const errData = await response.json();
                    errMsg = errData.error || errMsg;
                } catch (_) {}
                alert(`Export failed: ${errMsg}`);
                return;
            }

            // Convert response to Blob and trigger browser download
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = `session_${sessionId}_logs.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(blobUrl);
        } catch (error) {
            alert(`Export error: ${error.message}`);
        } finally {
            if (btn) { btn.innerHTML = origText; btn.disabled = false; }
        }
    },

    /**
     * Run calibration
     */
    async runCalibration() {
        const motorId = parseInt(document.getElementById('cal-motor').value);
        const pwmMin = parseInt(document.getElementById('cal-pwm-min').value);
        const pwmMax = parseInt(document.getElementById('cal-pwm-max').value);
        const rate = parseFloat(document.getElementById('cal-rate').value);

        // Get calibration profile config
        const response = await fetch('/api/current-control/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ motor_ids: [motorId], pwm_min: pwmMin, pwm_max: pwmMax, rate })
        });

        const data = await response.json();

        // Set up the profile for execution
        document.getElementById('profile-select').value = data.profile_config.profile;
        this.onProfileChange(data.profile_config.profile);

        // Fill in parameters
        for (const [key, value] of Object.entries(data.profile_config.params)) {
            const input = document.getElementById(`param-${key}`);
            if (input) input.value = value;
        }

        // Select only the calibration motor
        this.selectAllMotors(false);
        document.querySelector(`input[name="motor"][value="${motorId}"]`).checked = true;

        // Generate preview
        await this.generatePreview();

        document.getElementById('calibration-status').innerHTML = `
            <div class="result-success">
                <p>Calibration profile ready! Click "Execute Profile" to run the sweep.</p>
                <p>After completion, note the session ID and use "Process Calibration".</p>
            </div>
        `;

        // Switch to control tab
        this.switchTab('control');
    },

    /**
     * Process calibration
     */
    async processCalibration() {
        const sessionId = document.getElementById('cal-session').value;
        const motorId = parseInt(document.getElementById('process-motor').value);

        if (!sessionId) {
            alert('Please enter the session ID');
            return;
        }

        try {
            const response = await fetch('/api/current-control/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, motor_id: motorId })
            });

            const data = await response.json();

            if (response.ok) {
                alert(`Calibration processed! ${data.num_points} points from PWM ${data.pwm_range[0]} to ${data.pwm_range[1]}`);
                this.loadCalibrationCurve();
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Processing error:', error);
            alert('Error processing calibration: ' + error.message);
        }
    },

    /**
     * Load calibration curve
     */
    async loadCalibrationCurve() {
        const motorId = parseInt(document.getElementById('curve-motor').value);

        try {
            const response = await fetch(`/api/current-control/calibrations/${motorId}`);

            if (!response.ok) {
                const data = await response.json();
                this.currentChart.data.labels = [];
                this.currentChart.data.datasets[0].data = [];
                this.currentChart.update();
                return;
            }

            const data = await response.json();

            this.currentChart.data.labels = data.pwm;
            this.currentChart.data.datasets[0].data = data.current;
            this.currentChart.update();
        } catch (error) {
            console.error('Error loading curve:', error);
        }
    },

    /**
     * Look up PWM for target current
     */
    async lookupPWM() {
        const motorId = parseInt(document.getElementById('current-motor').value);
        const targetCurrent = parseFloat(document.getElementById('target-current').value);

        try {
            const response = await fetch('/api/current-control/lookup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ motor_id: motorId, target_current: targetCurrent })
            });

            const data = await response.json();
            const container = document.getElementById('lookup-result');

            if (response.ok) {
                container.className = 'result-success';
                container.innerHTML = `
                    <p><strong>Target:</strong> ${targetCurrent} A</p>
                    <p><strong>PWM:</strong> ${data.pwm}</p>
                    <p><strong>Expected Current:</strong> ${data.expected_current?.toFixed(2) || '--'} A</p>
                `;
            } else {
                container.className = 'result-error';
                container.innerHTML = `<strong>Error:</strong> ${data.error}`;
            }
        } catch (error) {
            console.error('Lookup error:', error);
        }
    }
};

/**
 * Sequence Builder for Multi-Sequence Profiles
 */
const SequenceBuilder = {
    sequences: [],
    app: null,  // Reference to DroneControl

    init(appRef) {
        this.app = appRef;
        this.sequences = [];
        this.attachEventListeners();
    },

    attachEventListeners() {
        const addBtn = document.getElementById('add-sequence-btn');
        if (addBtn) {
            addBtn.addEventListener('click', () => this.addSequence());
        }
    },

    /**
     * Snapshot all current DOM input values back into this.sequences[i].params
     * so they survive a re-render.
     */
    captureCurrentValues() {
        this.sequences.forEach((seq, index) => {
            const profile = this.app.profiles.find(p => p.name === seq.profile);
            if (!profile) return;
            for (const [key, param] of Object.entries(profile.parameters)) {
                if (param.type === 'array') continue;
                const input = document.getElementById(`seq-${index}-param-${key}`);
                if (!input) continue;
                if (input.type === 'checkbox') {
                    seq.params[key] = input.checked;
                } else {
                    const v = param.type === 'int'
                        ? parseInt(input.value)
                        : parseFloat(input.value);
                    seq.params[key] = isNaN(v) ? input.value : v;
                }
            }
        });
    },

    addSequence() {
        this.captureCurrentValues();
        const defaultProfile = this.app.profiles.find(p => p.name !== "Multi-Sequence");
        const profileName = defaultProfile ? defaultProfile.name : "Linear Ramp";
        this.sequences.push({
            profile: profileName,
            params: this.getDefaultParams(profileName)
        });
        this.render();
    },

    removeSequence(index) {
        this.captureCurrentValues();
        this.sequences.splice(index, 1);
        this.render();
    },

    moveSequence(index, direction) {
        this.captureCurrentValues();
        if (direction === 'up' && index > 0) {
            [this.sequences[index], this.sequences[index - 1]] =
            [this.sequences[index - 1], this.sequences[index]];
        } else if (direction === 'down' && index < this.sequences.length - 1) {
            [this.sequences[index], this.sequences[index + 1]] =
            [this.sequences[index + 1], this.sequences[index]];
        }
        this.render();
    },

    getDefaultParams(profileName) {
        const profile = this.app.profiles.find(p => p.name === profileName);
        if (!profile) return {};

        const params = {};
        for (const [key, param] of Object.entries(profile.parameters)) {
            if (param.default !== undefined) {
                params[key] = param.default;
            }
        }
        return params;
    },

    getProfileOptions(selectedProfile) {
        return this.app.profiles
            .filter(p => p.name !== "Multi-Sequence")  // Don't allow nested multi-sequences
            .map(p => `<option value="${p.name}" ${p.name === selectedProfile ? 'selected' : ''}>${p.name}</option>`)
            .join('');
    },

    onSequenceProfileChange(index, newProfileName) {
        this.sequences[index].profile = newProfileName;
        this.sequences[index].params = this.getDefaultParams(newProfileName);
        this.renderSequenceParams(index);
    },

    render() {
        const container = document.getElementById('sequences-container');
        if (!container) return;

        container.innerHTML = '';

        if (this.sequences.length === 0) {
            container.innerHTML = '<p class="placeholder" style="color: var(--text-muted); font-style: italic;">No sequences added yet. Click "Add Sequence" to start.</p>';
            return;
        }

        this.sequences.forEach((seq, index) => {
            const seqDiv = document.createElement('div');
            seqDiv.className = 'sequence-item';
            seqDiv.innerHTML = `
                <div class="sequence-header">
                    <h4>Sequence ${index + 1}</h4>
                    <div class="sequence-actions">
                        ${index > 0 ? '<button type="button" class="btn-icon" data-action="up" data-index="${index}" title="Move up">↑</button>' : ''}
                        ${index < this.sequences.length - 1 ? '<button type="button" class="btn-icon" data-action="down" data-index="${index}" title="Move down">↓</button>' : ''}
                        <button type="button" class="btn-danger btn-small" data-action="remove" data-index="${index}">Remove</button>
                    </div>
                </div>
                <div class="sequence-params">
                    <div class="form-group">
                        <label for="seq-${index}-profile">Profile Type</label>
                        <select id="seq-${index}-profile" class="form-control">
                            ${this.getProfileOptions(seq.profile)}
                        </select>
                    </div>
                    <div id="seq-${index}-params"></div>
                </div>
            `;
            container.appendChild(seqDiv);

            // Attach event listeners to buttons in this sequence
            seqDiv.querySelectorAll('[data-action]').forEach(btn => {
                const action = btn.dataset.action;
                const idx = parseInt(btn.dataset.index);
                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    if (action === 'remove') {
                        this.removeSequence(idx);
                    } else if (action === 'up') {
                        this.moveSequence(idx, 'up');
                    } else if (action === 'down') {
                        this.moveSequence(idx, 'down');
                    }
                });
            });

            // Attach profile change listener
            const profileSelect = document.getElementById(`seq-${index}-profile`);
            profileSelect.addEventListener('change', (e) => {
                this.onSequenceProfileChange(index, e.target.value);
            });

            // Render parameters for current profile
            this.renderSequenceParams(index);
        });
    },

    renderSequenceParams(index) {
        const seq = this.sequences[index];
        const profile = this.app.profiles.find(p => p.name === seq.profile);
        if (!profile) return;

        const container = document.getElementById(`seq-${index}-params`);
        if (!container) return;

        container.innerHTML = '';

        for (const [key, param] of Object.entries(profile.parameters)) {
            // Use id format "seq-{index}-param-{key}" so collectParams can find it
            const inputId = `seq-${index}-param-${key}`;

            if (param.type === 'array') {
                // Skip array (custom steps) inside multi-sequence for now
                continue;
            } else if (param.type === 'bool') {
                container.innerHTML += `
                    <div class="form-group">
                        <label for="${inputId}">${param.label || key}</label>
                        <input type="checkbox" id="${inputId}" ${param.default ? 'checked' : ''}>
                    </div>`;
            } else {
                const step = param.type === 'float' ? '0.1' : '1';
                const val = seq.params[key] !== undefined ? seq.params[key] : (param.default !== undefined ? param.default : '');
                container.innerHTML += `
                    <div class="form-group">
                        <label for="${inputId}">${param.label || key}</label>
                        <input type="number" id="${inputId}" class="form-control"
                               value="${val}"
                               min="${param.min !== undefined ? param.min : ''}"
                               max="${param.max !== undefined ? param.max : ''}"
                               step="${step}">
                    </div>`;
            }
        }
    },

    collectParams() {
        const sequences = [];

        this.sequences.forEach((seq, index) => {
            const profile = this.app.profiles.find(p => p.name === seq.profile);
            if (!profile) return;

            const params = {};

            // Collect parameters for this sequence
            for (const key of Object.keys(profile.parameters)) {
                const param = profile.parameters[key];

                if (param.type === 'array') {
                    // Special handling for array types (custom steps)
                    const steps = [];
                    const stepRows = document.querySelectorAll(`#seq-${index}-params .step-row`);
                    stepRows.forEach(row => {
                        const pwm = parseInt(row.querySelector('.step-pwm')?.value);
                        const duration = parseFloat(row.querySelector('.step-duration')?.value);
                        if (!isNaN(pwm) && !isNaN(duration)) {
                            steps.push({ pwm, duration_sec: duration });
                        }
                    });
                    params[key] = steps;
                } else {
                    const input = document.getElementById(`seq-${index}-param-${key}`);
                    if (input) {
                        if (input.type === 'checkbox') {
                            params[key] = input.checked;
                        } else if (param.type === 'int') {
                            params[key] = parseInt(input.value) || param.default || 0;
                        } else if (param.type === 'float') {
                            params[key] = parseFloat(input.value) || param.default || 0.0;
                        } else {
                            params[key] = input.value;
                        }
                    }
                }
            }

            sequences.push({
                profile: seq.profile,
                params: params
            });
        });

        const loopCount = parseInt(document.getElementById('loop-count')?.value) || 0;

        return {
            sequences: sequences,
            loop_count: loopCount
        };
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => DroneControl.init());
