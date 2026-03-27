package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that lands by commanding OFFBOARD descent velocity, then force-disarms.
 *
 * Stays in OFFBOARD mode (no mode switching needed) and sends:
 *   - XY position hold (current position)
 *   - Z velocity downward (controlled descent)
 *
 * Three-phase completion:
 *   1. DESCENDING — send downward velocity setpoints via OFFBOARD
 *   2. ON_GROUND — altitude < threshold, force disarm the vehicle
 *   3. DISARMED — confirm disarmed state
 */
public class LandStep extends TestStep {
    private static final double GROUND_ALTITUDE_THRESHOLD = 0.5;   // meters
    private static final double DESCENT_RATE = 0.7;                // m/s (NED positive = down)
    private static final double DEFAULT_SETTLE_SECONDS = 2.0;
    private static final long DISARM_RESEND_MS = 500;

    private enum Phase { DESCENDING, ON_GROUND, DISARMED }

    private Phase phase;
    private long settleStartTime;
    private double settleSeconds;
    private long lastCommandTime;
    private long lastDisarmTime;

    // Hold XY position during descent
    private double holdX;
    private double holdY;
    private boolean positionCaptured;

    public LandStep(double timeoutSeconds) {
        this(timeoutSeconds, DEFAULT_SETTLE_SECONDS);
    }

    public LandStep(double timeoutSeconds, double settleSeconds) {
        super("land", timeoutSeconds);
        this.settleSeconds = settleSeconds;
        this.phase = Phase.DESCENDING;
        this.settleStartTime = 0;
        this.lastCommandTime = 0;
        this.lastDisarmTime = 0;
        this.holdX = 0;
        this.holdY = 0;
        this.positionCaptured = false;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        phase = Phase.DESCENDING;
        settleStartTime = 0;
        positionCaptured = false;
        lastCommandTime = currentTime;
        lastDisarmTime = 0;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Capture XY position on first valid state
        if (!positionCaptured && state.hasPosition) {
            holdX = state.x;
            holdY = state.y;
            positionCaptured = true;
        }

        if (!positionCaptured) return;

        switch (phase) {
            case DESCENDING:
                // Send descent velocity setpoints at 10Hz (keeps OFFBOARD alive)
                if (currentTime - lastCommandTime > 100) {
                    commandSender.descendAtPosition(holdX, holdY, DESCENT_RATE);
                    lastCommandTime = currentTime;
                }
                break;

            case ON_GROUND:
                // Force disarm repeatedly until confirmed
                if (currentTime - lastDisarmTime > DISARM_RESEND_MS) {
                    commandSender.disarmForce();
                    lastDisarmTime = currentTime;
                }
                break;

            case DISARMED:
                // Nothing to do
                break;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasPosition) {
            return false;
        }

        double altitude = state.getAltitude();
        boolean disarmed = state.hasHeartbeat && !state.armed;
        long now = System.currentTimeMillis();

        switch (phase) {
            case DESCENDING:
                if (altitude < GROUND_ALTITUDE_THRESHOLD) {
                    phase = Phase.ON_GROUND;
                    settleStartTime = now;
                    lastDisarmTime = 0;  // trigger immediate disarm in next update
                    System.out.println(String.format(
                            " near ground (alt=%.2fm), force disarming...", altitude));
                }
                // If disarmed while still high, the vehicle is falling — NOT landed
                return false;

            case ON_GROUND:
                // Only accept disarm when near ground
                if (disarmed) {
                    phase = Phase.DISARMED;
                    markCompleted(String.format("alt=%.2fm, disarmed", altitude));
                    return true;
                }
                return false;

            case DISARMED:
                return false;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        if (!state.hasPosition) return null;
        double alt = state.getAltitude();
        double vz = state.getVerticalSpeed();
        return String.format("[land] alt=%.1fm vz=%.1fm/s armed=%s phase=%s",
                alt, vz, state.armed, phase);
    }
}
