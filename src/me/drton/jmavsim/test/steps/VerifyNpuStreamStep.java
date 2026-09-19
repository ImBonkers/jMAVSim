package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

import java.util.List;

/**
 * Step that verifies NPU NAMED_VALUE_FLOAT messages are being received.
 * Checks that hasNpu flag is set and values are non-zero for the full duration.
 */
public class VerifyNpuStreamStep extends TestStep {
    private double durationSeconds;
    private List<String> expectedFields;
    private long validSinceTime;

    public VerifyNpuStreamStep(double durationSeconds, List<String> expectedFields, double timeoutSeconds) {
        super("verifyNpuStream", timeoutSeconds);
        this.durationSeconds = durationSeconds;
        this.expectedFields = expectedFields;
        this.validSinceTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        validSinceTime = 0;
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        boolean valid = state.hasNpu;
        // Optionally check that values are non-zero (inference is running)
        if (valid && expectedFields.contains("npu_fps")) {
            valid = state.npuFps > 0;
        }

        long now = System.currentTimeMillis();
        if (valid) {
            if (validSinceTime == 0) {
                validSinceTime = now;
            }
            double held = (now - validSinceTime) / 1000.0;
            if (held >= durationSeconds) {
                markCompleted(String.format("npu_ms=%.1f npu_fps=%.1f npu_avg=%.1f",
                        state.npuMs, state.npuFps, state.npuAvg));
                return true;
            }
        } else {
            validSinceTime = 0;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        return String.format("[verifyNpuStream] hasNpu=%s fps=%.1f ms=%.1f elapsed=%.1fs",
                state.hasNpu, state.npuFps, state.npuMs, getElapsedSeconds());
    }

    @Override
    public String getDisplayName() {
        return "[verifyNpuStream]";
    }
}
