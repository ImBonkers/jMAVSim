package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands landing and waits for ground contact.
 *
 * Two-phase completion:
 *   1. DESCENDING — uses AUTO.LAND mode, waits for low altitude + low vz
 *   2. ON_GROUND — must remain on ground for settleSeconds to confirm
 *
 * Also confirms vehicle disarms after landing (PX4 auto-disarms on ground).
 */
public class LandStep extends TestStep {
    private static final double GROUND_ALTITUDE_THRESHOLD = 0.3;   // meters
    private static final double VERTICAL_VELOCITY_THRESHOLD = 0.15; // m/s
    private static final double DEFAULT_SETTLE_SECONDS = 2.0;
    private static final long MODE_RESEND_INTERVAL_MS = 2000;

    private enum Phase { DESCENDING, ON_GROUND }

    private Phase phase;
    private long settleStartTime;
    private double settleSeconds;
    private long lastCommandTime;
    private boolean landModeSet;

    public LandStep(double timeoutSeconds) {
        this(timeoutSeconds, DEFAULT_SETTLE_SECONDS);
    }

    public LandStep(double timeoutSeconds, double settleSeconds) {
        super("land", timeoutSeconds);
        this.settleSeconds = settleSeconds;
        this.phase = Phase.DESCENDING;
        this.settleStartTime = 0;
        this.lastCommandTime = 0;
        this.landModeSet = false;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        phase = Phase.DESCENDING;
        settleStartTime = 0;
        landModeSet = false;
        lastCommandTime = currentTime;

        // Use AUTO.LAND mode for proper landing behavior
        commandSender.setAutoLandMode();
        landModeSet = true;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Resend land mode periodically in case it wasn't accepted
        if (phase == Phase.DESCENDING &&
                currentTime - lastCommandTime > MODE_RESEND_INTERVAL_MS) {
            commandSender.setAutoLandMode();
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
        long now = System.currentTimeMillis();

        switch (phase) {
            case DESCENDING:
                if (altitude < GROUND_ALTITUDE_THRESHOLD &&
                        verticalSpeed < VERTICAL_VELOCITY_THRESHOLD) {
                    phase = Phase.ON_GROUND;
                    settleStartTime = now;
                    System.out.println(String.format(
                            " ground contact (alt=%.2fm), confirming...", altitude));
                }
                return false;

            case ON_GROUND:
                // If we bounced or drifted up, go back to descending
                if (altitude >= GROUND_ALTITUDE_THRESHOLD) {
                    phase = Phase.DESCENDING;
                    return false;
                }
                // Check if vehicle disarmed (PX4 auto-disarms on ground)
                if (!state.armed) {
                    markCompleted(String.format("alt=%.2fm, disarmed", altitude));
                    return true;
                }
                double settled = (now - settleStartTime) / 1000.0;
                if (settled >= settleSeconds) {
                    markCompleted(String.format("alt=%.2fm, on ground %.1fs (still armed)",
                            altitude, settled));
                    return true;
                }
                return false;
        }
        return false;
    }
}
