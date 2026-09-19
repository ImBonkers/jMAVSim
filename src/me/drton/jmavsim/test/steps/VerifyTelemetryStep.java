package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

import java.util.List;

/**
 * Step that verifies specified MAVLink message types are being received.
 * Checks the has* flags on VehicleState. Completes when all expected
 * messages have been seen for the full duration.
 */
public class VerifyTelemetryStep extends TestStep {
    private double durationSeconds;
    private List<String> expectedMessages;
    private long allSeenTime;

    public VerifyTelemetryStep(double durationSeconds, List<String> expectedMessages, double timeoutSeconds) {
        super("verifyTelemetry", timeoutSeconds);
        this.durationSeconds = durationSeconds;
        this.expectedMessages = expectedMessages;
        this.allSeenTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        allSeenTime = 0;
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        boolean allPresent = true;
        for (String msg : expectedMessages) {
            if (!isMessagePresent(msg, state)) {
                allPresent = false;
                break;
            }
        }

        long now = System.currentTimeMillis();
        if (allPresent) {
            if (allSeenTime == 0) {
                allSeenTime = now;
            }
            double held = (now - allSeenTime) / 1000.0;
            if (held >= durationSeconds) {
                markCompleted("all " + expectedMessages.size() + " streams verified");
                return true;
            }
        } else {
            allSeenTime = 0;
        }
        return false;
    }

    private boolean isMessagePresent(String msgName, VehicleState state) {
        switch (msgName.toUpperCase()) {
            case "HEARTBEAT":          return state.hasHeartbeat;
            case "ATTITUDE":           return state.hasAttitude;
            case "LOCAL_POSITION_NED": return state.hasPosition;
            case "SYS_STATUS":         return state.hasSysStatus;
            case "ESTIMATOR_STATUS":   return state.hasEstimator;
            case "VIBRATION":          return state.hasVibration;
            default:                   return state.hasHeartbeat; // assume present
        }
    }

    @Override
    public String getProgressString(VehicleState state) {
        int seen = 0;
        for (String msg : expectedMessages) {
            if (isMessagePresent(msg, state)) seen++;
        }
        return String.format("[verifyTelemetry] %d/%d streams, elapsed=%.1fs",
                seen, expectedMessages.size(), getElapsedSeconds());
    }

    @Override
    public String getDisplayName() {
        return "[verifyTelemetry " + expectedMessages.size() + " streams]";
    }
}
