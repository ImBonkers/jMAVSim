package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that waits for the vehicle to disarm itself (e.g. after RTL/auto-land).
 * Does NOT send disarm commands — just monitors.
 */
public class WaitForDisarmStep extends TestStep {

    public WaitForDisarmStep(double timeoutSeconds) {
        super("waitForDisarm", timeoutSeconds);
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasHeartbeat) return false;
        if (!state.armed) {
            markCompleted(null);
            return true;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        if (!state.hasPosition) return null;
        return String.format("[waitForDisarm] alt=%.1fm armed=%s elapsed=%.1fs",
                state.getAltitude(), state.armed, getElapsedSeconds());
    }

    @Override
    public String getDisplayName() {
        return "[waitForDisarm]";
    }
}
