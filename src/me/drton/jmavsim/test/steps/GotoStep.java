package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands vehicle to a position and waits for arrival.
 * Completes when distance to target is within tolerance and speed is low.
 */
public class GotoStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 1000;
    private static final double SPEED_THRESHOLD = 0.5;  // m/s

    private double targetX;
    private double targetY;
    private double targetZ;
    private double tolerance;
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
        super("goto", timeoutSeconds);
        this.targetX = x;
        this.targetY = y;
        this.targetZ = z;
        this.tolerance = tolerance;
        this.lastCommandTime = 0;
    }

    private boolean offboardSet = false;

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        offboardSet = false;
        // Send position setpoint first (required before OFFBOARD mode)
        commandSender.gotoPosition(targetX, targetY, targetZ);
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Send position setpoints at high rate for OFFBOARD mode
        if (currentTime - lastCommandTime > 100) {  // 10Hz
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

        // Check if at target with low speed
        if (distance <= tolerance && speed < SPEED_THRESHOLD) {
            markCompleted(String.format("err=%.2fm", distance));
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return String.format("[goto %.0f,%.0f,%.0f]", targetX, targetY, targetZ);
    }
}
