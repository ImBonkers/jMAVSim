package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that waits for a specified duration.
 * Used for delays between operations or waiting for stabilization.
 */
public class WaitStep extends TestStep {
    private double durationSeconds;

    public WaitStep(double durationSeconds) {
        super("wait", durationSeconds + 1.0);  // Timeout slightly longer than duration
        this.durationSeconds = durationSeconds;
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        double elapsed = getElapsedSeconds();
        if (elapsed >= durationSeconds) {
            markCompleted(null);
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return String.format("[wait %.1fs]", durationSeconds);
    }
}
