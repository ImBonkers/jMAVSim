package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands vehicle to a position and waits for arrival.
 *
 * Two-phase completion:
 *   1. TRANSIT — moving toward target, not yet within tolerance
 *   2. SETTLING — position must stay within tolerance with low speed for settleSeconds
 *
 * If position drifts out of tolerance during settling, resets back to transit.
 */
public class GotoStep extends TestStep {
    private static final double SPEED_THRESHOLD = 0.5;  // m/s
    private static final double DEFAULT_SETTLE_SECONDS = 2.0;

    private enum Phase { TRANSIT, SETTLING }

    private double targetX;
    private double targetY;
    private double targetZ;
    private double tolerance;
    private double settleSeconds;

    private Phase phase;
    private long settleStartTime;
    private boolean offboardSet;
    private long lastCommandTime;

    /**
     * Create goto step
     * @param x Target X position (North) in meters
     * @param y Target Y position (East) in meters
     * @param z Target Z position (Down) in meters - negative is up!
     * @param tolerance Distance tolerance in meters
     * @param timeoutSeconds Timeout in seconds
     */
    public GotoStep(double x, double y, double z, double tolerance, double timeoutSeconds) {
        this(x, y, z, tolerance, timeoutSeconds, DEFAULT_SETTLE_SECONDS);
    }

    public GotoStep(double x, double y, double z, double tolerance, double timeoutSeconds,
                    double settleSeconds) {
        super("goto", timeoutSeconds);
        this.targetX = x;
        this.targetY = y;
        this.targetZ = z;
        this.tolerance = tolerance;
        this.settleSeconds = settleSeconds;
        this.phase = Phase.TRANSIT;
        this.settleStartTime = 0;
        this.offboardSet = false;
        this.lastCommandTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        phase = Phase.TRANSIT;
        offboardSet = false;
        settleStartTime = 0;
        // Send position setpoint first (required before OFFBOARD mode)
        commandSender.gotoPosition(targetX, targetY, targetZ);
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Send position setpoints at 10Hz for OFFBOARD mode
        if (currentTime - lastCommandTime > 100) {
            commandSender.gotoPosition(targetX, targetY, targetZ);
            lastCommandTime = currentTime;

            // Set OFFBOARD mode after a few setpoints sent
            if (!offboardSet && getElapsedSeconds() > 0.5) {
                commandSender.setOffboardMode();
                offboardSet = true;
            }
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasPosition) {
            return false;
        }

        double distance = state.distanceTo(targetX, targetY, targetZ);
        double speed = state.getSpeed();
        long now = System.currentTimeMillis();

        switch (phase) {
            case TRANSIT:
                if (distance <= tolerance && speed < SPEED_THRESHOLD) {
                    phase = Phase.SETTLING;
                    settleStartTime = now;
                }
                return false;

            case SETTLING:
                if (distance > tolerance || speed >= SPEED_THRESHOLD) {
                    // Drifted out — reset
                    phase = Phase.TRANSIT;
                    return false;
                }
                double settled = (now - settleStartTime) / 1000.0;
                if (settled >= settleSeconds) {
                    markCompleted(String.format("err=%.2fm, settled=%.1fs",
                            distance, settled));
                    return true;
                }
                return false;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        if (!state.hasPosition) return null;
        double dist = state.distanceTo(targetX, targetY, targetZ);
        double spd = state.getSpeed();
        return String.format("[goto] dist=%.1fm spd=%.1fm/s phase=%s",
                dist, spd, phase);
    }

    @Override
    public String getDisplayName() {
        return String.format("[goto %.0f,%.0f,%.0f]", targetX, targetY, targetZ);
    }
}
