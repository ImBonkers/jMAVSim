package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that waits for the vehicle to become armed.
 * Sends arm command and waits for armed state in HEARTBEAT.
 */
public class ArmStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 1000;
    private long lastCommandTime;

    public ArmStep(double timeoutSeconds) {
        super("arm", timeoutSeconds);
        this.lastCommandTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        commandSender.arm();
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Only resend arm command if not yet armed
        if (!state.armed && currentTime - lastCommandTime > COMMAND_RESEND_INTERVAL_MS) {
            commandSender.arm();
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasHeartbeat) {
            return false;
        }

        if (state.armed) {
            markCompleted(null);
            return true;
        }
        return false;
    }
}
