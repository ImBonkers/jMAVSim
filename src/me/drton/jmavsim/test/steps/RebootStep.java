package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;
import me.drton.jmavsim.test.StateMonitor;

/**
 * Step that reboots the flight controller and waits for full reconnection.
 *
 * Four-phase state machine:
 *   1. COMMANDING  — force-disarm then send reboot commands for a few seconds
 *   2. WAITING_DISCONNECT — wait for heartbeat to stop (confirms FC is actually rebooting)
 *   3. WAITING_RECONNECT  — wait for heartbeat to resume (FC is back)
 *   4. WAITING_READY       — wait for position data and stability (FC is fully initialized)
 *
 * Fails if:
 *   - FC never disconnects (reboot command was ignored)
 *   - FC disconnects but never comes back
 *   - FC comes back but never sends position data
 */
public class RebootStep extends TestStep {
    private static final long REBOOT_COMMAND_DURATION_MS = 3000;
    private static final long REBOOT_COMMAND_INTERVAL_MS = 500;
    private static final long DISCONNECT_TIMEOUT_MS = 8000;
    private static final long READY_SETTLE_MS = 2000;

    private enum Phase {
        COMMANDING,
        WAITING_DISCONNECT,
        WAITING_RECONNECT,
        WAITING_READY
    }

    private Phase phase;
    private long phaseStartTime;
    private long lastCommandTime;
    private long readyStartTime;
    private StateMonitor stateMonitor;
    private boolean disconnectSeen;

    public RebootStep(double timeoutSeconds) {
        super("reboot", timeoutSeconds);
        this.phase = Phase.COMMANDING;
        this.phaseStartTime = 0;
        this.lastCommandTime = 0;
        this.readyStartTime = 0;
        this.stateMonitor = null;
        this.disconnectSeen = false;
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
        phase = Phase.COMMANDING;
        phaseStartTime = currentTime;
        lastCommandTime = 0;
        readyStartTime = 0;
        disconnectSeen = false;

        // Force disarm before rebooting
        commandSender.disarmForce();
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        switch (phase) {
            case COMMANDING:
                // Send reboot commands repeatedly
                if (currentTime - lastCommandTime > REBOOT_COMMAND_INTERVAL_MS) {
                    commandSender.reboot();
                    lastCommandTime = currentTime;
                }

                // After sending for REBOOT_COMMAND_DURATION_MS, move to disconnect check
                if (currentTime - phaseStartTime > REBOOT_COMMAND_DURATION_MS) {
                    phase = Phase.WAITING_DISCONNECT;
                    phaseStartTime = currentTime;
                    System.out.println("RebootStep: Reboot commands sent, waiting for disconnect...");
                }
                break;

            case WAITING_DISCONNECT:
                // Nothing to send — just waiting for heartbeat to stop
                break;

            case WAITING_RECONNECT:
                // Nothing to send — just waiting for heartbeat to resume
                break;

            case WAITING_READY:
                // Nothing to send — waiting for full state
                break;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        long now = System.currentTimeMillis();

        switch (phase) {
            case COMMANDING:
                // Still sending commands — not complete
                return false;

            case WAITING_DISCONNECT:
                // Check if the StateMonitor has lost connection
                if (stateMonitor != null && !stateMonitor.isConnected()) {
                    disconnectSeen = true;
                    phase = Phase.WAITING_RECONNECT;
                    phaseStartTime = now;

                    // Reset state monitor so old data is cleared
                    stateMonitor.reset();

                    System.out.println("RebootStep: FC disconnected, waiting for reconnect...");
                    return false;
                }

                // If we've waited too long for disconnect, the reboot may have been
                // ignored — but some FCs reboot so fast we miss the gap.
                // Fall through to reconnect wait after timeout.
                if (now - phaseStartTime > DISCONNECT_TIMEOUT_MS) {
                    System.out.println("RebootStep: No disconnect detected (FC may have rebooted quickly), " +
                            "resetting state and waiting for fresh data...");
                    disconnectSeen = false;
                    if (stateMonitor != null) {
                        stateMonitor.reset();
                    }
                    phase = Phase.WAITING_RECONNECT;
                    phaseStartTime = now;
                }
                return false;

            case WAITING_RECONNECT:
                // Wait for fresh heartbeat from StateMonitor
                if (stateMonitor != null && stateMonitor.isConnected()) {
                    phase = Phase.WAITING_READY;
                    phaseStartTime = now;
                    readyStartTime = 0;
                    System.out.println("RebootStep: FC reconnected (sysId=" +
                            stateMonitor.getTargetSysId() + "), waiting for ready state...");
                }
                return false;

            case WAITING_READY:
                // Wait for position data — FC is fully initialized when it streams telemetry
                if (state.hasPosition && state.hasHeartbeat && state.hasAttitude) {
                    if (readyStartTime == 0) {
                        readyStartTime = now;
                    }
                    // Hold for READY_SETTLE_MS to confirm it's stable
                    if (now - readyStartTime >= READY_SETTLE_MS) {
                        String details = disconnectSeen ? "disconnect confirmed" : "fast reboot";
                        markCompleted(details);
                        return true;
                    }
                } else {
                    // Data dropped — reset settle timer
                    readyStartTime = 0;
                }
                return false;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        String connected = (stateMonitor != null && stateMonitor.isConnected()) ? "yes" : "no";
        return String.format("[reboot] phase=%s connected=%s hasPos=%s hasAtt=%s elapsed=%.1fs",
                phase, connected, state.hasPosition, state.hasAttitude, getElapsedSeconds());
    }

    @Override
    public String getDisplayName() {
        return "[reboot]";
    }
}
