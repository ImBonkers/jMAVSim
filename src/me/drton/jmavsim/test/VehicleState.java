package me.drton.jmavsim.test;

/**
 * Data class holding telemetry snapshot from the vehicle.
 * Contains position, velocity, attitude, and armed state.
 */
public class VehicleState {
    // Position in NED frame (meters)
    public double x;
    public double y;
    public double z;  // negative is up

    // Velocity in NED frame (m/s)
    public double vx;
    public double vy;
    public double vz;

    // Attitude (radians)
    public double roll;
    public double pitch;
    public double yaw;

    // Armed state
    public boolean armed;

    // Flight mode
    public int baseMode;
    public int customMode;

    // Timestamp (simulation time in ms)
    public long timestamp;

    // Data validity flags
    public boolean hasPosition;
    public boolean hasAttitude;
    public boolean hasHeartbeat;

    public VehicleState() {
        this.x = 0;
        this.y = 0;
        this.z = 0;
        this.vx = 0;
        this.vy = 0;
        this.vz = 0;
        this.roll = 0;
        this.pitch = 0;
        this.yaw = 0;
        this.armed = false;
        this.baseMode = 0;
        this.customMode = 0;
        this.timestamp = 0;
        this.hasPosition = false;
        this.hasAttitude = false;
        this.hasHeartbeat = false;
    }

    /**
     * Copy constructor
     */
    public VehicleState(VehicleState other) {
        this.x = other.x;
        this.y = other.y;
        this.z = other.z;
        this.vx = other.vx;
        this.vy = other.vy;
        this.vz = other.vz;
        this.roll = other.roll;
        this.pitch = other.pitch;
        this.yaw = other.yaw;
        this.armed = other.armed;
        this.baseMode = other.baseMode;
        this.customMode = other.customMode;
        this.timestamp = other.timestamp;
        this.hasPosition = other.hasPosition;
        this.hasAttitude = other.hasAttitude;
        this.hasHeartbeat = other.hasHeartbeat;
    }

    /**
     * Get altitude (positive up) from NED z coordinate
     */
    public double getAltitude() {
        return -z;
    }

    /**
     * Get ground speed (horizontal)
     */
    public double getGroundSpeed() {
        return Math.sqrt(vx * vx + vy * vy);
    }

    /**
     * Get total speed (3D)
     */
    public double getSpeed() {
        return Math.sqrt(vx * vx + vy * vy + vz * vz);
    }

    /**
     * Get vertical speed (positive up)
     */
    public double getVerticalSpeed() {
        return -vz;
    }

    /**
     * Get horizontal distance from origin
     */
    public double getHorizontalDistanceFromOrigin() {
        return Math.sqrt(x * x + y * y);
    }

    /**
     * Get distance to a target position
     */
    public double distanceTo(double tx, double ty, double tz) {
        double dx = x - tx;
        double dy = y - ty;
        double dz = z - tz;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Get horizontal distance to a target position
     */
    public double horizontalDistanceTo(double tx, double ty) {
        double dx = x - tx;
        double dy = y - ty;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Check if data is valid for flight operations
     */
    public boolean isValid() {
        return hasPosition && hasHeartbeat;
    }

    @Override
    public String toString() {
        return String.format(
            "VehicleState[pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f) att=(%.1f,%.1f,%.1f) armed=%s]",
            x, y, z, vx, vy, vz,
            Math.toDegrees(roll), Math.toDegrees(pitch), Math.toDegrees(yaw),
            armed
        );
    }
}
