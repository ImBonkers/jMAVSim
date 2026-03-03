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

    public CommandSender(MAVLinkSchema schema, int sysId, int componentId) {
        super(schema, sysId, componentId);
        this.targetSysId = 1;  // Default autopilot system ID
        this.targetCompId = 1; // Default autopilot component ID
        this.commandSequence = 0;
        // Disable heartbeat from command sender
        setHeartbeatInterval(0);
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
        System.out.println(String.format("CommandSender: Sending GOTO position (%.2f, %.2f, %.2f)", x, y, z));
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

        // Handle command acknowledgments if needed
        if ("COMMAND_ACK".equals(msg.getMsgName())) {
            int command = msg.getInt("command");
            int result = msg.getInt("result");
            if (result == 0) {
                // MAV_RESULT_ACCEPTED
            } else {
                System.err.println("CommandSender: Command " + command + " rejected with result " + result);
            }
        }
    }
}
