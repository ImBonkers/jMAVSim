package me.drton.jmavsim.test;

import me.drton.jmavlib.mavlink.MAVLinkMessage;
import me.drton.jmavlib.mavlink.MAVLinkSchema;
import me.drton.jmavsim.MAVLinkSystem;

/**
 * MAVLink system that sends commands to the flight controller.
 * Provides methods for arm, disarm, takeoff, land, and goto operations.
 */
public class CommandSender extends MAVLinkSystem {
    private static final int MAV_CMD_COMPONENT_ARM_DISARM = 400;
    private static final int MAV_CMD_NAV_TAKEOFF = 22;
    private static final int MAV_CMD_NAV_LAND = 21;
    private static final int MAV_CMD_DO_SET_MODE = 176;
    private static final int MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN = 246;
    private static final int MAV_CMD_SET_MESSAGE_INTERVAL = 511;
    private static final int MAV_CMD_DO_ORBIT = 34;

    // PX4 custom_mode encoding: (sub_mode << 24) | (main_mode << 16)
    private static final int PX4_MAIN_MODE_MANUAL = 1;
    private static final int PX4_MAIN_MODE_ALTITUDE = 2;
    private static final int PX4_MAIN_MODE_POSITION = 3;
    private static final int PX4_MAIN_MODE_AUTO = 4;
    private static final int PX4_MAIN_MODE_OFFBOARD = 6;
    private static final int PX4_SUB_MODE_AUTO_READY = 1;
    private static final int PX4_SUB_MODE_AUTO_TAKEOFF = 2;
    private static final int PX4_SUB_MODE_AUTO_LOITER = 3;
    private static final int PX4_SUB_MODE_AUTO_MISSION = 4;
    private static final int PX4_SUB_MODE_AUTO_RTL = 5;
    private static final int PX4_SUB_MODE_AUTO_LAND = 6;

    // SET_POSITION_TARGET_LOCAL_NED type mask bits
    private static final int POSITION_TARGET_TYPEMASK_X_IGNORE = 1;
    private static final int POSITION_TARGET_TYPEMASK_Y_IGNORE = 2;
    private static final int POSITION_TARGET_TYPEMASK_Z_IGNORE = 4;
    private static final int POSITION_TARGET_TYPEMASK_VX_IGNORE = 8;
    private static final int POSITION_TARGET_TYPEMASK_VY_IGNORE = 16;
    private static final int POSITION_TARGET_TYPEMASK_VZ_IGNORE = 32;
    private static final int POSITION_TARGET_TYPEMASK_AX_IGNORE = 64;
    private static final int POSITION_TARGET_TYPEMASK_AY_IGNORE = 128;
    private static final int POSITION_TARGET_TYPEMASK_AZ_IGNORE = 256;
    private static final int POSITION_TARGET_TYPEMASK_FORCE_SET = 512;
    private static final int POSITION_TARGET_TYPEMASK_YAW_IGNORE = 1024;
    private static final int POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE = 2048;

    private int targetSysId;
    private int targetCompId;
    private int commandSequence;

    // Last COMMAND_ACK received
    private volatile int lastAckCommand;
    private volatile int lastAckResult;

    // Last PARAM_VALUE received
    private volatile String lastParamId;
    private volatile float lastParamValue;

    public CommandSender(MAVLinkSchema schema, int sysId, int componentId) {
        super(schema, sysId, componentId);
        this.targetSysId = 1;  // Default autopilot system ID
        this.targetCompId = 1; // Default autopilot component ID
        this.commandSequence = 0;
        this.lastAckCommand = -1;
        this.lastAckResult = -1;
        this.lastParamId = null;
        this.lastParamValue = 0;
        // Disable heartbeat from command sender
        setHeartbeatInterval(0);
    }

    /**
     * Check if a COMMAND_ACK was received for the given command ID with MAV_RESULT_ACCEPTED.
     * Clears the stored ACK after reading (consume-once).
     */
    /**
     * Get the last received param ID, or null.
     */
    public String getLastParamId() {
        return lastParamId;
    }

    /**
     * Get the last received param value.
     */
    public float getLastParamValue() {
        return lastParamValue;
    }

    /**
     * Clear the last param value (consume-once pattern).
     */
    public void clearLastParam() {
        lastParamId = null;
    }

    public boolean consumeAck(int commandId) {
        if (lastAckCommand == commandId && lastAckResult == 0) {
            lastAckCommand = -1;
            lastAckResult = -1;
            return true;
        }
        return false;
    }

    /**
     * Request a specific MAVLink message at a given interval.
     * @param msgId MAVLink message ID
     * @param intervalUs Interval in microseconds (-1 to disable, 0 for default)
     */
    public void requestMessageInterval(int msgId, long intervalUs) {
        sendCommandLong(MAV_CMD_SET_MESSAGE_INTERVAL,
                (float) msgId, (float) intervalUs, 0, 0, 0, 0, 0);
    }

    /**
     * Set the target system and component ID for commands.
     */
    public void setTarget(int sysId, int compId) {
        this.targetSysId = sysId;
        this.targetCompId = compId;
    }

    /**
     * Set target system ID only
     */
    public void setTargetSysId(int sysId) {
        this.targetSysId = sysId;
    }

    /**
     * Send reboot command to flight controller
     */
    public void reboot() {
        // param1=1 reboot autopilot, param2=0 don't reboot companion
        // Send with confirmation=1 to ensure it's processed
        MAVLinkMessage msg = new MAVLinkMessage(schema, "COMMAND_LONG",
                sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("target_component", targetCompId);
        msg.set("command", MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN);
        msg.set("confirmation", 1);
        msg.set("param1", 1.0f);  // Reboot autopilot
        msg.set("param2", 0.0f);
        msg.set("param3", 0.0f);
        msg.set("param4", 0.0f);
        msg.set("param5", 0.0f);
        msg.set("param6", 0.0f);
        msg.set("param7", 0.0f);
        sendMessage(msg);
        System.out.println("CommandSender: Sending REBOOT command");
    }

    /**
     * Send arm command
     */
    public void arm() {
        sendCommandLong(MAV_CMD_COMPONENT_ARM_DISARM, 1, 0, 0, 0, 0, 0, 0);
        System.out.println("CommandSender: Sending ARM command");
    }

    /**
     * Send disarm command
     */
    public void disarm() {
        sendCommandLong(MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0);
        System.out.println("CommandSender: Sending DISARM command");
    }

    /**
     * Send disarm command with force flag
     */
    public void disarmForce() {
        sendCommandLong(MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196, 0, 0, 0, 0, 0);
        System.out.println("CommandSender: Sending FORCE DISARM command");
    }

    /**
     * Send takeoff command
     * @param altitude Target altitude in meters (positive up)
     */
    public void takeoff(double altitude) {
        // param7 is altitude for NAV_TAKEOFF
        sendCommandLong(MAV_CMD_NAV_TAKEOFF, 0, 0, 0, Float.NaN, Float.NaN, Float.NaN, (float) altitude);
        System.out.println("CommandSender: Sending TAKEOFF command to altitude " + altitude + "m");
    }

    /**
     * Set flight mode to OFFBOARD
     * Uses SET_MODE message for more reliable mode switching
     */
    public void setOffboardMode() {
        // PX4 OFFBOARD mode: main_mode = 6, sub_mode = 0
        // custom_mode = (sub_mode << 24) | (main_mode << 16) = 6 << 16 = 393216
        MAVLinkMessage msg = new MAVLinkMessage(schema, "SET_MODE", sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("base_mode", 209);  // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | ARMED | HIL
        msg.set("custom_mode", 6 << 16);  // OFFBOARD
        sendMessage(msg);
        System.out.println("CommandSender: Setting OFFBOARD mode");
    }

    /**
     * Set flight mode to AUTO TAKEOFF
     */
    public void setAutoTakeoffMode() {
        // PX4: main_mode=AUTO(4), sub_mode=TAKEOFF(2)
        MAVLinkMessage msg = new MAVLinkMessage(schema, "SET_MODE", sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("base_mode", 209);
        msg.set("custom_mode", (2 << 24) | (4 << 16));
        sendMessage(msg);
        System.out.println("CommandSender: Setting AUTO.TAKEOFF mode");
    }

    /**
     * Set flight mode to AUTO LAND
     */
    public void setAutoLandMode() {
        // PX4: main_mode=AUTO(4), sub_mode=LAND(6)
        MAVLinkMessage msg = new MAVLinkMessage(schema, "SET_MODE", sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("base_mode", 209);
        msg.set("custom_mode", (6 << 24) | (4 << 16));
        sendMessage(msg);
        System.out.println("CommandSender: Setting AUTO.LAND mode");
    }

    /**
     * Set flight mode by name.
     * Supported: MANUAL, ALTITUDE, POSITION, OFFBOARD, MISSION, RTL, LAND, TAKEOFF
     */
    public void setMode(String modeName) {
        int customMode;
        switch (modeName.toUpperCase()) {
            case "MANUAL":
                customMode = PX4_MAIN_MODE_MANUAL << 16;
                break;
            case "ALTITUDE":
                customMode = PX4_MAIN_MODE_ALTITUDE << 16;
                break;
            case "POSITION":
                customMode = PX4_MAIN_MODE_POSITION << 16;
                break;
            case "OFFBOARD":
                customMode = PX4_MAIN_MODE_OFFBOARD << 16;
                break;
            case "MISSION":
                customMode = (PX4_SUB_MODE_AUTO_MISSION << 24) | (PX4_MAIN_MODE_AUTO << 16);
                break;
            case "RTL":
                customMode = (PX4_SUB_MODE_AUTO_RTL << 24) | (PX4_MAIN_MODE_AUTO << 16);
                break;
            case "LAND":
                customMode = (PX4_SUB_MODE_AUTO_LAND << 24) | (PX4_MAIN_MODE_AUTO << 16);
                break;
            case "TAKEOFF":
                customMode = (PX4_SUB_MODE_AUTO_TAKEOFF << 24) | (PX4_MAIN_MODE_AUTO << 16);
                break;
            case "LOITER":
                customMode = (PX4_SUB_MODE_AUTO_LOITER << 24) | (PX4_MAIN_MODE_AUTO << 16);
                break;
            default:
                System.err.println("CommandSender: Unknown mode: " + modeName);
                return;
        }
        MAVLinkMessage msg = new MAVLinkMessage(schema, "SET_MODE", sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("base_mode", 209);  // MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | ARMED | HIL
        msg.set("custom_mode", customMode);
        sendMessage(msg);
        System.out.println("CommandSender: Setting mode " + modeName);
    }

    /**
     * Set a PX4 parameter via PARAM_SET message.
     * @param paramId Parameter name (max 16 chars)
     * @param value Parameter value
     */
    public void setParam(String paramId, float value) {
        setParam(paramId, value, 9);  // MAV_PARAM_TYPE_REAL32
    }

    public void setParamInt(String paramId, int value) {
        // PX4 encodes INT32 params as float with bit-reinterpretation
        float encoded = Float.intBitsToFloat(value);
        setParam(paramId, encoded, 6);  // MAV_PARAM_TYPE_INT32
    }

    public void setParam(String paramId, float value, int paramType) {
        MAVLinkMessage msg = new MAVLinkMessage(schema, "PARAM_SET",
                sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("target_component", targetCompId);
        msg.set("param_id", paramId);
        msg.set("param_value", value);
        msg.set("param_type", paramType);
        sendMessage(msg);
        System.out.println("CommandSender: Setting param " + paramId + " = " + value + " (type=" + paramType + ")");
    }

    /**
     * Request a parameter value via PARAM_REQUEST_READ.
     */
    public void requestParam(String paramId) {
        MAVLinkMessage msg = new MAVLinkMessage(schema, "PARAM_REQUEST_READ",
                sysId, componentId, protocolVersion);
        msg.set("target_system", targetSysId);
        msg.set("target_component", targetCompId);
        msg.set("param_id", paramId);
        msg.set("param_index", -1);  // Use name, not index
        sendMessage(msg);
    }

    /**
     * Send orbit command (MAV_CMD_DO_ORBIT).
     * @param radius Orbit radius in meters (positive = CW, negative = CCW)
     * @param velocity Tangential velocity in m/s
     * @param centerX Center X (North) in local NED, NaN to orbit current position
     * @param centerY Center Y (East) in local NED
     * @param centerZ Center Z (Down) in local NED
     */
    public void orbit(float radius, float velocity, float centerX, float centerY, float centerZ) {
        sendCommandLong(MAV_CMD_DO_ORBIT, radius, velocity, 0, Float.NaN, centerX, centerY, centerZ);
        System.out.println("CommandSender: Sending ORBIT command r=" + radius + " v=" + velocity);
    }

    /**
     * Send land command
     */
    public void land() {
        sendCommandLong(MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0);
        System.out.println("CommandSender: Sending LAND command");
    }

    /**
     * Send position setpoint in local NED frame
     * @param x North position in meters
     * @param y East position in meters
     * @param z Down position in meters (negative is up)
     */
    public void gotoPosition(double x, double y, double z) {
        gotoPosition(x, y, z, Float.NaN);
    }

    /**
     * Send position setpoint in local NED frame with yaw
     * @param x North position in meters
     * @param y East position in meters
     * @param z Down position in meters (negative is up)
     * @param yaw Yaw angle in radians (NaN to ignore)
     */
    public void gotoPosition(double x, double y, double z, float yaw) {
        MAVLinkMessage msg = new MAVLinkMessage(schema, "SET_POSITION_TARGET_LOCAL_NED",
                sysId, componentId, protocolVersion);

        // Set time boot (not critical)
        msg.set("time_boot_ms", 0);

        // Target system/component
        msg.set("target_system", targetSysId);
        msg.set("target_component", targetCompId);

        // Coordinate frame: MAV_FRAME_LOCAL_NED = 1
        msg.set("coordinate_frame", 1);

        // Type mask: use position, ignore velocity and acceleration
        int typeMask = POSITION_TARGET_TYPEMASK_VX_IGNORE |
                       POSITION_TARGET_TYPEMASK_VY_IGNORE |
                       POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                       POSITION_TARGET_TYPEMASK_AX_IGNORE |
                       POSITION_TARGET_TYPEMASK_AY_IGNORE |
                       POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                       POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE;

        if (Float.isNaN(yaw)) {
            typeMask |= POSITION_TARGET_TYPEMASK_YAW_IGNORE;
        }

        msg.set("type_mask", typeMask);

        // Position
        msg.set("x", (float) x);
        msg.set("y", (float) y);
        msg.set("z", (float) z);

        // Velocity (ignored)
        msg.set("vx", 0.0f);
        msg.set("vy", 0.0f);
        msg.set("vz", 0.0f);

        // Acceleration (ignored)
        msg.set("afx", 0.0f);
        msg.set("afy", 0.0f);
        msg.set("afz", 0.0f);

        // Yaw
        if (!Float.isNaN(yaw)) {
            msg.set("yaw", yaw);
        } else {
            msg.set("yaw", 0.0f);
        }
        msg.set("yaw_rate", 0.0f);

        sendMessage(msg);
    }

    /**
     * Send velocity setpoint in local NED frame while holding XY position.
     * Used for controlled descent — stays in OFFBOARD mode.
     * @param holdX North position to hold (meters)
     * @param holdY East position to hold (meters)
     * @param vz Down velocity (positive = descending) in m/s
     */
    public void descendAtPosition(double holdX, double holdY, double vz) {
        MAVLinkMessage msg = new MAVLinkMessage(schema, "SET_POSITION_TARGET_LOCAL_NED",
                sysId, componentId, protocolVersion);

        msg.set("time_boot_ms", 0);
        msg.set("target_system", targetSysId);
        msg.set("target_component", targetCompId);
        msg.set("coordinate_frame", 1);  // MAV_FRAME_LOCAL_NED

        // Use XY position + Z velocity: ignore Z position, VX, VY, all accel, yaw
        int typeMask = POSITION_TARGET_TYPEMASK_Z_IGNORE |
                       POSITION_TARGET_TYPEMASK_VX_IGNORE |
                       POSITION_TARGET_TYPEMASK_VY_IGNORE |
                       POSITION_TARGET_TYPEMASK_AX_IGNORE |
                       POSITION_TARGET_TYPEMASK_AY_IGNORE |
                       POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                       POSITION_TARGET_TYPEMASK_YAW_IGNORE |
                       POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE;

        msg.set("type_mask", typeMask);
        msg.set("x", (float) holdX);
        msg.set("y", (float) holdY);
        msg.set("z", 0.0f);       // ignored
        msg.set("vx", 0.0f);      // ignored
        msg.set("vy", 0.0f);      // ignored
        msg.set("vz", (float) vz); // descent rate
        msg.set("afx", 0.0f);
        msg.set("afy", 0.0f);
        msg.set("afz", 0.0f);
        msg.set("yaw", 0.0f);
        msg.set("yaw_rate", 0.0f);

        sendMessage(msg);
    }

    /**
     * Send a COMMAND_LONG message
     */
    private void sendCommandLong(int command, float param1, float param2, float param3,
                                  float param4, float param5, float param6, float param7) {
        MAVLinkMessage msg = new MAVLinkMessage(schema, "COMMAND_LONG",
                sysId, componentId, protocolVersion);

        msg.set("target_system", targetSysId);
        msg.set("target_component", targetCompId);
        msg.set("command", command);
        msg.set("confirmation", commandSequence++);
        msg.set("param1", param1);
        msg.set("param2", param2);
        msg.set("param3", param3);
        msg.set("param4", param4);
        msg.set("param5", param5);
        msg.set("param6", param6);
        msg.set("param7", param7);

        sendMessage(msg);
    }

    @Override
    public void handleMessage(MAVLinkMessage msg) {
        super.handleMessage(msg);

        // Handle command acknowledgments
        if ("COMMAND_ACK".equals(msg.getMsgName())) {
            int command = msg.getInt("command");
            int result = msg.getInt("result");
            lastAckCommand = command;
            lastAckResult = result;
            if (result != 0) {
                System.err.println("CommandSender: Command " + command + " rejected with result " + result);
            }
        } else if ("PARAM_VALUE".equals(msg.getMsgName())) {
            lastParamId = msg.getString("param_id").trim();
            lastParamValue = msg.getFloat("param_value");
        }
    }
}
