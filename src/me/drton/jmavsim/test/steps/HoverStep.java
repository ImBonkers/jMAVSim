package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that validates position stability over a duration.
 * Completes when position drift stays below threshold for the entire duration.
 * Keeps sending position setpoints to maintain OFFBOARD mode.
 */
public class HoverStep extends TestStep {
    private double durationSeconds;
    private double maxDrift;

    // Reference position at start of hover
    private double refX;
    private double refY;
    private double refZ;
    private boolean refSet;

    // Tracking
    private long hoverStartTime;
    private double maxDriftObserved;
    private long lastCommandTime;

    public HoverStep(double durationSeconds, double maxDrift, double timeoutSeconds) {
        super("hover", timeoutSeconds);
        this.durationSeconds = durationSeconds;
        this.maxDrift = maxDrift;
        this.refSet = false;
        this.hoverStartTime = 0;
        this.maxDriftObserved = 0;
        this.lastCommandTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        refSet = false;
        hoverStartTime = 0;
        maxDriftObserved = 0;
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Keep sending position setpoints to maintain OFFBOARD mode
        if (refSet && currentTime - lastCommandTime > 100) {  // 10Hz
            commandSender.gotoPosition(refX, refY, refZ);
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasPosition) {
            return false;
        }

        long currentTime = System.currentTimeMillis();

        // Set reference position on first valid state
        if (!refSet) {
            refX = state.x;
            refY = state.y;
            refZ = state.z;
            refSet = true;
            hoverStartTime = currentTime;
            maxDriftObserved = 0;
            return false;
        }

        // Calculate current drift from reference
        double dx = state.x - refX;
        double dy = state.y - refY;
        double dz = state.z - refZ;
        double drift = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Update max drift observed
        if (drift > maxDriftObserved) {
            maxDriftObserved = drift;
        }

        // Check if drift exceeds threshold - reset timer
        if (drift > maxDrift) {
            // Reset: use current position as new reference
            refX = state.x;
            refY = state.y;
            refZ = state.z;
            hoverStartTime = currentTime;
            maxDriftObserved = 0;
            return false;
        }

        // Check if we've maintained position for the required duration
        double hoverTime = (currentTime - hoverStartTime) / 1000.0;
        if (hoverTime >= durationSeconds) {
            markCompleted(String.format("drift=%.2fm", maxDriftObserved));
            return true;
        }

        return false;
    }

    @Override
    public String getDisplayName() {
        return String.format("[hover %.1fs]", durationSeconds);
    }
}
