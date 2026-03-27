package me.drton.jmavsim.test;

/**
 * Data class holding telemetry snapshot from the vehicle.
 * Contains position, velocity, attitude, armed state, and FC diagnostics.
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

    // --- Flight controller diagnostics (from SYS_STATUS) ---
    // CPU load: 0-1000 (0.1% increments, so 1000 = 100%)
    public int cpuLoad;
    // Communication drop rate (0.01% increments)
    public int commDropRate;
    // Communication errors count
    public int commErrors;
    // Autopilot-specific error counters
    public int errorsCount1;
    public int errorsCount2;
    public int errorsCount3;
    public int errorsCount4;

    // --- EKF status (from ESTIMATOR_STATUS) ---
    public int ekfFlags;
    public float ekfVelRatio;
    public float ekfPosHorizRatio;
    public float ekfPosVertRatio;
    public float ekfMagRatio;
    public float ekfPosHorizAccuracy;
    public float ekfPosVertAccuracy;

    // --- Vibration (from VIBRATION) ---
    public float vibrationX;
    public float vibrationY;
    public float vibrationZ;
    public long clipping0;
    public long clipping1;
    public long clipping2;

    // --- FC boot time (from SYSTEM_TIME) ---
    public long bootTimeMs;

    // --- Attitude setpoint (from ATTITUDE_TARGET) ---
    public double rollSetpoint;
    public double pitchSetpoint;
    public double yawSetpoint;
    public double thrustSetpoint;
    public boolean hasAttitudeTarget;

    // --- NPU stats (from NAMED_VALUE_FLOAT) ---
    public float npuMs;        // inference latency in ms
    public float npuFps;       // inferences per second
    public float npuAvg;       // average inference time
    public boolean hasNpu;

    // Data validity flags
    public boolean hasPosition;
    public boolean hasAttitude;
    public boolean hasHeartbeat;
    public boolean hasSysStatus;
    public boolean hasEstimator;
    public boolean hasVibration;

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
        this.cpuLoad = 0;
        this.commDropRate = 0;
        this.commErrors = 0;
        this.errorsCount1 = 0;
        this.errorsCount2 = 0;
        this.errorsCount3 = 0;
        this.errorsCount4 = 0;
        this.ekfFlags = 0;
        this.ekfVelRatio = 0;
        this.ekfPosHorizRatio = 0;
        this.ekfPosVertRatio = 0;
        this.ekfMagRatio = 0;
        this.ekfPosHorizAccuracy = 0;
        this.ekfPosVertAccuracy = 0;
        this.vibrationX = 0;
        this.vibrationY = 0;
        this.vibrationZ = 0;
        this.clipping0 = 0;
        this.clipping1 = 0;
        this.clipping2 = 0;
        this.bootTimeMs = 0;
        this.rollSetpoint = 0;
        this.pitchSetpoint = 0;
        this.yawSetpoint = 0;
        this.thrustSetpoint = 0;
        this.hasAttitudeTarget = false;
        this.npuMs = 0;
        this.npuFps = 0;
        this.npuAvg = 0;
        this.hasNpu = false;
        this.hasPosition = false;
        this.hasAttitude = false;
        this.hasHeartbeat = false;
        this.hasSysStatus = false;
        this.hasEstimator = false;
        this.hasVibration = false;
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
        this.cpuLoad = other.cpuLoad;
        this.commDropRate = other.commDropRate;
        this.commErrors = other.commErrors;
        this.errorsCount1 = other.errorsCount1;
        this.errorsCount2 = other.errorsCount2;
        this.errorsCount3 = other.errorsCount3;
        this.errorsCount4 = other.errorsCount4;
        this.ekfFlags = other.ekfFlags;
        this.ekfVelRatio = other.ekfVelRatio;
        this.ekfPosHorizRatio = other.ekfPosHorizRatio;
        this.ekfPosVertRatio = other.ekfPosVertRatio;
        this.ekfMagRatio = other.ekfMagRatio;
        this.ekfPosHorizAccuracy = other.ekfPosHorizAccuracy;
        this.ekfPosVertAccuracy = other.ekfPosVertAccuracy;
        this.vibrationX = other.vibrationX;
        this.vibrationY = other.vibrationY;
        this.vibrationZ = other.vibrationZ;
        this.clipping0 = other.clipping0;
        this.clipping1 = other.clipping1;
        this.clipping2 = other.clipping2;
        this.bootTimeMs = other.bootTimeMs;
        this.rollSetpoint = other.rollSetpoint;
        this.pitchSetpoint = other.pitchSetpoint;
        this.yawSetpoint = other.yawSetpoint;
        this.thrustSetpoint = other.thrustSetpoint;
        this.hasAttitudeTarget = other.hasAttitudeTarget;
        this.npuMs = other.npuMs;
        this.npuFps = other.npuFps;
        this.npuAvg = other.npuAvg;
        this.hasNpu = other.hasNpu;
        this.hasPosition = other.hasPosition;
        this.hasAttitude = other.hasAttitude;
        this.hasHeartbeat = other.hasHeartbeat;
        this.hasSysStatus = other.hasSysStatus;
        this.hasEstimator = other.hasEstimator;
        this.hasVibration = other.hasVibration;
    }

    /**
     * Get CPU load as percentage (0.0 - 100.0)
     */
    public double getCpuPercent() {
        return cpuLoad / 10.0;
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
