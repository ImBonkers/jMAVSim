package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that sets the NPU load percentage on the companion board.
 * Sends a custom parameter or NAMED_VALUE_INT to control NPU utilization.
 * Completes immediately after sending.
 */
public class SetNpuLoadStep extends TestStep {
    private int percent;

    public SetNpuLoadStep(int percent, double timeoutSeconds) {
        super("setNpuLoad", timeoutSeconds);
        this.percent = percent;
        this.critical = false;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        // Set NPU load via param — board-specific parameter
        commandSender.setParam("NPU_DUTY_PCT", (float) percent);
        System.out.println("SetNpuLoadStep: NPU load set to " + percent + "%");
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (getElapsedSeconds() >= 1.0) {
            markCompleted("npu=" + percent + "%");
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return "[setNpuLoad " + percent + "%]";
    }
}
