package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that switches the flight mode and waits for confirmation via HEARTBEAT custom_mode.
 * Matches on main_mode (bits 16-23) and optionally sub_mode (bits 24-31).
 */
public class SetModeStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 1000;

    private String mode;
    private int expectedMainMode;
    private int expectedSubMode;  // -1 = don't check sub_mode
    private long lastCommandTime;

    public SetModeStep(String mode, double timeoutSeconds) {
        super("setMode", timeoutSeconds);
        this.mode = mode.toUpperCase();
        this.lastCommandTime = 0;
        computeExpected(this.mode);
    }

    private void computeExpected(String mode) {
        switch (mode) {
            case "MANUAL":    expectedMainMode = 1; expectedSubMode = -1; break;
            case "ALTITUDE":  expectedMainMode = 2; expectedSubMode = -1; break;
            case "POSITION":  expectedMainMode = 3; expectedSubMode = -1; break;
            case "OFFBOARD":  expectedMainMode = 6; expectedSubMode = -1; break;
            case "MISSION":   expectedMainMode = 4; expectedSubMode = 4;  break;
            case "RTL":       expectedMainMode = 4; expectedSubMode = 5;  break;
            case "LAND":      expectedMainMode = 4; expectedSubMode = 6;  break;
            case "TAKEOFF":   expectedMainMode = 4; expectedSubMode = 2;  break;
            case "LOITER":    expectedMainMode = 4; expectedSubMode = 3;  break;
            default:          expectedMainMode = -1; expectedSubMode = -1; break;
        }
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        commandSender.setMode(mode);
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        if (currentTime - lastCommandTime > COMMAND_RESEND_INTERVAL_MS) {
            commandSender.setMode(mode);
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (!state.hasHeartbeat) return false;

        // Extract main_mode and sub_mode from PX4 custom_mode
        int mainMode = (state.customMode >> 16) & 0xFF;
        int subMode = (state.customMode >> 24) & 0xFF;

        boolean mainMatch = (mainMode == expectedMainMode);
        boolean subMatch = (expectedSubMode < 0) || (subMode == expectedSubMode);

        if (mainMatch && subMatch) {
            markCompleted(String.format("mode=%s (main=%d sub=%d)", mode, mainMode, subMode));
            return true;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        if (!state.hasHeartbeat) return null;
        int mainMode = (state.customMode >> 16) & 0xFF;
        int subMode = (state.customMode >> 24) & 0xFF;
        return String.format("[setMode %s] current main=%d sub=%d, want main=%d sub=%s",
                mode, mainMode, subMode, expectedMainMode,
                expectedSubMode < 0 ? "*" : String.valueOf(expectedSubMode));
    }

    @Override
    public String getDisplayName() {
        return "[setMode " + mode + "]";
    }
}
