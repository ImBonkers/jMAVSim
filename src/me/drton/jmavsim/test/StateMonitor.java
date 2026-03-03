package me.drton.jmavsim.test;

import me.drton.jmavlib.mavlink.MAVLinkMessage;
import me.drton.jmavlib.mavlink.MAVLinkSchema;
import me.drton.jmavsim.MAVLinkNode;

/**
 * MAVLink node that monitors incoming messages to track vehicle state.
 * Listens for HEARTBEAT, LOCAL_POSITION_NED, and ATTITUDE messages.
 */
public class StateMonitor extends MAVLinkNode {
    private final VehicleState currentState;
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

        if ("HEARTBEAT".equals(msgName)) {
            handleHeartbeat(msg);
        } else if ("LOCAL_POSITION_NED".equals(msgName)) {
            handleLocalPositionNed(msg);
        } else if ("ATTITUDE".equals(msgName)) {
            handleAttitude(msg);
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
