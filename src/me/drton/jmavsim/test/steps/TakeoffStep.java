package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands takeoff and waits for target altitude.
 * Completes when altitude is within tolerance and vertical velocity is low.
 */
public class TakeoffStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 2000;
    private static final double VERTICAL_VELOCITY_THRESHOLD = 0.5;  // m/s

    private double targetAltitude;
    private double tolerance;
    private long lastCommandTime;

    public TakeoffStep(double altitude, double tolerance, double timeoutSeconds) {
        super("takeoff", timeoutSeconds);
        this.targetAltitude = altitude;
        this.tolerance = tolerance;
        this.lastCommandTime = 0;
    }

    private boolean offboardSet = false;
    private double startX = 0;
    private double startY = 0;

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        offboardSet = false;
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Capture start position on first update with valid state
        if (!offboardSet && state.hasPosition) {
            startX = state.x;
            startY = state.y;
        }

        // Send position setpoints at high rate for OFFBOARD mode
        if (currentTime - lastCommandTime > 100) {  // 10Hz
            // Command current XY position, target altitude (negative Z in NED)
            commandSender.gotoPosition(startX, startY, -targetAltitude);
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

        double currentAltitude = state.getAltitude();
        double altitudeError = Math.abs(currentAltitude - targetAltitude);
        double verticalSpeed = Math.abs(state.getVerticalSpeed());

        // Check if at target altitude with low vertical velocity
        if (altitudeError <= tolerance && verticalSpeed < VERTICAL_VELOCITY_THRESHOLD) {
            markCompleted(String.format("alt=%.2fm", currentAltitude));
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return String.format("[takeoff %.1fm]", targetAltitude);
    }
}
