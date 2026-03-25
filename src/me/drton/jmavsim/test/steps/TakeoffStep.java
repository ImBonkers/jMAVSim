package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands takeoff and waits for target altitude.
 *
 * Three-phase completion:
 *   1. LIFTING — wait until altitude > liftoffThreshold (confirms vehicle left ground)
 *   2. CLIMBING — wait until altitude is within tolerance of target
 *   3. SETTLING — altitude must stay within tolerance with low vz for settleSeconds
 *
 * Fails immediately if timeout expires at any phase.
 */
public class TakeoffStep extends TestStep {
    private static final double VERTICAL_VELOCITY_THRESHOLD = 0.5;  // m/s
    private static final double DEFAULT_LIFTOFF_THRESHOLD = 0.5;    // meters
    private static final double DEFAULT_SETTLE_SECONDS = 2.0;

    private enum Phase { LIFTING, CLIMBING, SETTLING }

    private double targetAltitude;
    private double tolerance;
    private double liftoffThreshold;
    private double settleSeconds;

    private Phase phase;
    private long settleStartTime;
    private boolean offboardSet;
    private double startX;
    private double startY;
    private long lastCommandTime;

    public TakeoffStep(double altitude, double tolerance, double timeoutSeconds) {
        this(altitude, tolerance, timeoutSeconds, DEFAULT_SETTLE_SECONDS);
    }

    public TakeoffStep(double altitude, double tolerance, double timeoutSeconds,
                       double settleSeconds) {
        super("takeoff", timeoutSeconds);
        this.targetAltitude = altitude;
        this.tolerance = tolerance;
        this.liftoffThreshold = DEFAULT_LIFTOFF_THRESHOLD;
        this.settleSeconds = settleSeconds;
        this.phase = Phase.LIFTING;
        this.settleStartTime = 0;
        this.offboardSet = false;
        this.startX = 0;
        this.startY = 0;
        this.lastCommandTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        phase = Phase.LIFTING;
        offboardSet = false;
        settleStartTime = 0;
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Capture start position on first update with valid state
        if (!offboardSet && state.hasPosition) {
            startX = state.x;
            startY = state.y;
        }

        // Send position setpoints at 10Hz for OFFBOARD mode
        if (currentTime - lastCommandTime > 100) {
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
        long now = System.currentTimeMillis();

        switch (phase) {
            case LIFTING:
                // Phase 1: Confirm vehicle has left the ground
                if (currentAltitude > liftoffThreshold) {
                    phase = Phase.CLIMBING;
                    System.out.println(String.format(
                            " liftoff detected (alt=%.2fm)", currentAltitude));
                }
                return false;

            case CLIMBING:
                // Phase 2: Wait for altitude to be within tolerance
                if (altitudeError <= tolerance && verticalSpeed < VERTICAL_VELOCITY_THRESHOLD) {
                    phase = Phase.SETTLING;
                    settleStartTime = now;
                    System.out.println(String.format(
                            " at target altitude (alt=%.2fm), settling...", currentAltitude));
                }
                return false;

            case SETTLING:
                // Phase 3: Confirm altitude stays within tolerance for settleSeconds
                if (altitudeError > tolerance || verticalSpeed >= VERTICAL_VELOCITY_THRESHOLD) {
                    // Fell out of tolerance — go back to climbing
                    phase = Phase.CLIMBING;
                    return false;
                }
                double settled = (now - settleStartTime) / 1000.0;
                if (settled >= settleSeconds) {
                    markCompleted(String.format("alt=%.2fm, settled=%.1fs",
                            currentAltitude, settled));
                    return true;
                }
                return false;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return String.format("[takeoff %.1fm]", targetAltitude);
    }
}
