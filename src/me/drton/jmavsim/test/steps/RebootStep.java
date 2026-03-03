package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;
import me.drton.jmavsim.test.StateMonitor;

/**
 * Step that reboots the flight controller and waits for it to reconnect.
 * Sends reboot command for 2 seconds, then waits for reconnection.
 */
public class RebootStep extends TestStep {
    private static final long REBOOT_COMMAND_DURATION_MS = 2000;  // Send reboots for 2 seconds
    private static final long MIN_RECONNECT_TIME_MS = 5000;       // Wait at least 5 seconds total

    private long rebootPhaseEndTime;
    private boolean rebootPhaseDone;
    private long lastCommandTime;
    private StateMonitor stateMonitor;

    public RebootStep(double timeoutSeconds) {
        super("reboot", timeoutSeconds);
        this.rebootPhaseEndTime = 0;
        this.rebootPhaseDone = false;
        this.lastCommandTime = 0;
        this.stateMonitor = null;
    }

    /**
     * Set the state monitor reference for resetting state.
     */
    public void setStateMonitor(StateMonitor monitor) {
        this.stateMonitor = monitor;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        // First disarm if armed
        commandSender.disarmForce();
        rebootPhaseEndTime = currentTime + REBOOT_COMMAND_DURATION_MS;
        rebootPhaseDone = false;
        lastCommandTime = 0;

        // Reset state monitor immediately
        if (stateMonitor != null) {
            stateMonitor.reset();
        }
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Phase 1: Send reboot commands for REBOOT_COMMAND_DURATION_MS
        if (!rebootPhaseDone) {
            if (currentTime < rebootPhaseEndTime) {
                // Send reboot command every 500ms
                if (currentTime - lastCommandTime > 500) {
                    commandSender.reboot();
                    lastCommandTime = currentTime;
                }
            } else {
                // Done sending reboots
                rebootPhaseDone = true;
                System.out.println("RebootStep: Reboot commands sent, waiting for FC to reconnect...");

                // Reset state monitor to clear old state
                if (stateMonitor != null) {
                    stateMonitor.reset();
                }
            }
        }
        // Phase 2: Just wait for reconnection - don't send anything
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        // Only check for completion after reboot phase is done
        if (!rebootPhaseDone) {
            return false;
        }

        // Wait for fresh heartbeat after minimum time
        if (state.hasHeartbeat && getElapsedSeconds() > (MIN_RECONNECT_TIME_MS / 1000.0)) {
            System.out.println("RebootStep: Flight controller reconnected");
            markCompleted(null);
            return true;
        }

        return false;
    }

    @Override
    public String getDisplayName() {
        return "[reboot]";
    }
}
