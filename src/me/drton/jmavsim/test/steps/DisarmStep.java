package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that waits for the vehicle to become disarmed.
 * Sends disarm command and waits for disarmed state in HEARTBEAT.
 */
public class DisarmStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 1000;
    private long lastCommandTime;
    private boolean forceDisarm;

    public DisarmStep(double timeoutSeconds) {
        this(timeoutSeconds, false);
    }

    public DisarmStep(double timeoutSeconds, boolean forceDisarm) {
        super("disarm", timeoutSeconds);
        this.lastCommandTime = 0;
        this.forceDisarm = forceDisarm;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        if (forceDisarm) {
            commandSender.disarmForce();
        } else {
            commandSender.disarm();
        }
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Resend disarm command periodically
        if (currentTime - lastCommandTime > COMMAND_RESEND_INTERVAL_MS) {
            if (forceDisarm) {
                commandSender.disarmForce();
            } else {
                commandSender.disarm();
            }
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasHeartbeat) {
            return false;
        }

        // Check armed bit (bit 7) in base_mode - should be 0 for disarmed
        boolean armed = (state.baseMode & 0x80) != 0;
        if (!armed) {
            markCompleted(null);
            return true;
        }
        return false;
    }
}
