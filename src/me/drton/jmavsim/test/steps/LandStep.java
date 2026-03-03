package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands landing and waits for ground contact.
 * Completes when altitude is very low and vertical velocity is minimal.
 */
public class LandStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 2000;
    private static final double GROUND_ALTITUDE_THRESHOLD = 0.15;  // meters
    private static final double VERTICAL_VELOCITY_THRESHOLD = 0.1;  // m/s

    private long lastCommandTime;

    public LandStep(double timeoutSeconds) {
        super("land", timeoutSeconds);
        this.lastCommandTime = 0;
    }

    private double startX = 0;
    private double startY = 0;
    private boolean positionCaptured = false;

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        positionCaptured = false;
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Capture position on first update with valid state
        if (!positionCaptured && state.hasPosition) {
            startX = state.x;
            startY = state.y;
            positionCaptured = true;
        }

        // Send position setpoints at high rate - descend to ground
        if (currentTime - lastCommandTime > 100) {  // 10Hz
            // Command current XY position, ground level (Z=0 in NED)
            commandSender.gotoPosition(startX, startY, 0);
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasPosition) {
            return false;
        }

        double altitude = state.getAltitude();
        double verticalSpeed = Math.abs(state.getVerticalSpeed());

        // Check if on ground
        if (altitude < GROUND_ALTITUDE_THRESHOLD && verticalSpeed < VERTICAL_VELOCITY_THRESHOLD) {
            markCompleted(null);
            return true;
        }
        return false;
    }
}
