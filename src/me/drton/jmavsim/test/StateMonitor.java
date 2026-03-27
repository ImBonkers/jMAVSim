package me.drton.jmavsim.test;

import me.drton.jmavlib.mavlink.MAVLinkMessage;
import me.drton.jmavlib.mavlink.MAVLinkSchema;
import me.drton.jmavsim.MAVLinkNode;

/**
 * MAVLink node that monitors incoming messages to track vehicle state.
 * Listens for HEARTBEAT, LOCAL_POSITION_NED, ATTITUDE, SYS_STATUS,
 * ESTIMATOR_STATUS, VIBRATION, and SYSTEM_TIME messages.
 */
public class StateMonitor extends MAVLinkNode {
    private final VehicleState currentState;
    private StreamRateMonitor rateMonitor;
    private int targetSysId;
    private boolean connected;
    private long lastHeartbeatTime;
    private static final long HEARTBEAT_TIMEOUT_MS = 3000;

    public StateMonitor(MAVLinkSchema schema) {
        super(schema);
        this.currentState = new VehicleState();
        this.targetSysId = -1;  // Auto-detect
        this.connected = false;
        this.lastHeartbeatTime = 0;
    }

    public void setRateMonitor(StreamRateMonitor monitor) {
        this.rateMonitor = monitor;
    }

    /**
     * Set the target system ID to monitor.
     * Use -1 for auto-detect (will use first heartbeat received).
     */
    public void setTargetSysId(int sysId) {
        this.targetSysId = sysId;
    }

    /**
     * Get the current system ID being monitored.
     */
    public int getTargetSysId() {
        return targetSysId;
    }

    /**
     * Check if we're connected to a vehicle (receiving heartbeats).
     */
    public boolean isConnected() {
        return connected;
    }

    /**
     * Get a snapshot of the current vehicle state.
     * Returns a copy to avoid race conditions.
     */
    public synchronized VehicleState getState() {
        return new VehicleState(currentState);
    }

    /**
     * Reset the monitor state (for reboot detection).
     */
    public synchronized void reset() {
        currentState.hasHeartbeat = false;
        currentState.hasPosition = false;
        currentState.hasAttitude = false;
        currentState.hasSysStatus = false;
        currentState.hasEstimator = false;
        currentState.hasVibration = false;
        currentState.timestamp = 0;
        connected = false;
        lastHeartbeatTime = 0;
        System.out.println("StateMonitor: State reset");
    }

    /**
     * Get direct reference to current state (for performance).
     * Only use if you know what you're doing.
     */
    public VehicleState getCurrentStateRef() {
        return currentState;
    }

    @Override
    public synchronized void handleMessage(MAVLinkMessage msg) {
        String msgName = msg.getMsgName();

        // Auto-detect system ID from first heartbeat
        if (targetSysId < 0 && "HEARTBEAT".equals(msgName)) {
            targetSysId = msg.systemID;
            System.out.println("StateMonitor: Auto-detected system ID: " + targetSysId);
        }

        // Filter messages by system ID if set
        if (targetSysId > 0 && msg.systemID != targetSysId) {
            return;
        }

        // Track message rate
        if (rateMonitor != null) {
            rateMonitor.messageReceived(msg.getMsgType());
        }

        if ("HEARTBEAT".equals(msgName)) {
            handleHeartbeat(msg);
        } else if ("LOCAL_POSITION_NED".equals(msgName)) {
            handleLocalPositionNed(msg);
        } else if ("ATTITUDE".equals(msgName)) {
            handleAttitude(msg);
        } else if ("SYS_STATUS".equals(msgName)) {
            handleSysStatus(msg);
        } else if ("ATTITUDE_TARGET".equals(msgName)) {
            handleAttitudeTarget(msg);
        } else if ("ESTIMATOR_STATUS".equals(msgName)) {
            handleEstimatorStatus(msg);
        } else if ("VIBRATION".equals(msgName)) {
            handleVibration(msg);
        } else if ("SYSTEM_TIME".equals(msgName)) {
            handleSystemTime(msg);
        } else if ("NAMED_VALUE_FLOAT".equals(msgName)) {
            handleNamedValueFloat(msg);
        } else if ("STATUSTEXT".equals(msgName)) {
            handleStatusText(msg);
        }
    }

    private void handleStatusText(MAVLinkMessage msg) {
        try {
            String text = msg.getString("text");
            int severity = msg.getInt("severity");
            String severityStr;
            switch (severity) {
                case 0: severityStr = "EMERGENCY"; break;
                case 1: severityStr = "ALERT"; break;
                case 2: severityStr = "CRITICAL"; break;
                case 3: severityStr = "ERROR"; break;
                case 4: severityStr = "WARNING"; break;
                case 5: severityStr = "NOTICE"; break;
                case 6: severityStr = "INFO"; break;
                default: severityStr = "DEBUG"; break;
            }
            System.out.println("[FC " + severityStr + "] " + text);
        } catch (Exception e) {
            // Ignore parse errors
        }
    }

    private void handleHeartbeat(MAVLinkMessage msg) {
        int baseMode = msg.getInt("base_mode");
        int customMode = msg.getInt("custom_mode");

        synchronized (this) {
            currentState.baseMode = baseMode;
            currentState.customMode = customMode;
            // Armed state is bit 7 (0x80) of base_mode
            currentState.armed = (baseMode & 0x80) != 0;
            currentState.hasHeartbeat = true;
            currentState.timestamp = System.currentTimeMillis();
            lastHeartbeatTime = currentState.timestamp;
            connected = true;
        }
    }

    private void handleLocalPositionNed(MAVLinkMessage msg) {
        synchronized (this) {
            currentState.x = msg.getDouble("x");
            currentState.y = msg.getDouble("y");
            currentState.z = msg.getDouble("z");
            currentState.vx = msg.getDouble("vx");
            currentState.vy = msg.getDouble("vy");
            currentState.vz = msg.getDouble("vz");
            currentState.hasPosition = true;
            currentState.timestamp = System.currentTimeMillis();
        }
    }

    private void handleAttitude(MAVLinkMessage msg) {
        synchronized (this) {
            currentState.roll = msg.getDouble("roll");
            currentState.pitch = msg.getDouble("pitch");
            currentState.yaw = msg.getDouble("yaw");
            currentState.hasAttitude = true;
            currentState.timestamp = System.currentTimeMillis();
        }
    }

    private void handleAttitudeTarget(MAVLinkMessage msg) {
        synchronized (this) {
            // quaternion [w, x, y, z]
            float[] q = new float[4];
            q[0] = msg.getFloat("q[0]");  // w
            q[1] = msg.getFloat("q[1]");  // x
            q[2] = msg.getFloat("q[2]");  // y
            q[3] = msg.getFloat("q[3]");  // z
            // Convert quaternion to euler
            currentState.rollSetpoint = Math.atan2(2.0 * (q[0]*q[1] + q[2]*q[3]), 1.0 - 2.0 * (q[1]*q[1] + q[2]*q[2]));
            currentState.pitchSetpoint = Math.asin(Math.max(-1, Math.min(1, 2.0 * (q[0]*q[2] - q[3]*q[1]))));
            currentState.yawSetpoint = Math.atan2(2.0 * (q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3]));
            currentState.thrustSetpoint = msg.getFloat("thrust");
            currentState.hasAttitudeTarget = true;
        }
    }

    private void handleSysStatus(MAVLinkMessage msg) {
        synchronized (this) {
            currentState.cpuLoad = msg.getInt("load");
            currentState.commDropRate = msg.getInt("drop_rate_comm");
            currentState.commErrors = msg.getInt("errors_comm");
            currentState.errorsCount1 = msg.getInt("errors_count1");
            currentState.errorsCount2 = msg.getInt("errors_count2");
            currentState.errorsCount3 = msg.getInt("errors_count3");
            currentState.errorsCount4 = msg.getInt("errors_count4");
            currentState.hasSysStatus = true;
        }
    }

    private void handleEstimatorStatus(MAVLinkMessage msg) {
        synchronized (this) {
            currentState.ekfFlags = msg.getInt("flags");
            currentState.ekfVelRatio = msg.getFloat("vel_ratio");
            currentState.ekfPosHorizRatio = msg.getFloat("pos_horiz_ratio");
            currentState.ekfPosVertRatio = msg.getFloat("pos_vert_ratio");
            currentState.ekfMagRatio = msg.getFloat("mag_ratio");
            currentState.ekfPosHorizAccuracy = msg.getFloat("pos_horiz_accuracy");
            currentState.ekfPosVertAccuracy = msg.getFloat("pos_vert_accuracy");
            currentState.hasEstimator = true;
        }
    }

    private void handleVibration(MAVLinkMessage msg) {
        synchronized (this) {
            currentState.vibrationX = msg.getFloat("vibration_x");
            currentState.vibrationY = msg.getFloat("vibration_y");
            currentState.vibrationZ = msg.getFloat("vibration_z");
            currentState.clipping0 = msg.getLong("clipping_0");
            currentState.clipping1 = msg.getLong("clipping_1");
            currentState.clipping2 = msg.getLong("clipping_2");
            currentState.hasVibration = true;
        }
    }

    private void handleNamedValueFloat(MAVLinkMessage msg) {
        try {
            String name = msg.getString("name").trim();
            float value = msg.getFloat("value");
            synchronized (this) {
                switch (name) {
                    case "npu_ms":
                        currentState.npuMs = value;
                        currentState.hasNpu = true;
                        break;
                    case "npu_fps":
                        currentState.npuFps = value;
                        currentState.hasNpu = true;
                        break;
                    case "npu_avg":
                        currentState.npuAvg = value;
                        currentState.hasNpu = true;
                        break;
                }
            }
        } catch (Exception e) {
            // Ignore parse errors
        }
    }

    private void handleSystemTime(MAVLinkMessage msg) {
        synchronized (this) {
            currentState.bootTimeMs = msg.getLong("time_boot_ms");
        }
    }

    @Override
    public void update(long t, boolean paused) {
        // Check for connection timeout
        if (connected && lastHeartbeatTime > 0) {
            long now = System.currentTimeMillis();
            if (now - lastHeartbeatTime > HEARTBEAT_TIMEOUT_MS) {
                connected = false;
                System.err.println("StateMonitor: Connection lost (heartbeat timeout)");
            }
        }
    }
}
