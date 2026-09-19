package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that triggers a battery failsafe by setting SIM_BAT_DRAIN param
 * to rapidly drain the battery. Completes once the param is sent.
 */
public class SimulateBatteryFailsafeStep extends TestStep {
    private int voltagePercent;

    public SimulateBatteryFailsafeStep(int voltagePercent, double timeoutSeconds) {
        super("simulateBatteryFailsafe", timeoutSeconds);
        this.voltagePercent = voltagePercent;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        // Set simulated battery level to trigger failsafe
        // SIM_BAT_MIN_PCT sets the reported percentage
        commandSender.setParam("SIM_BAT_MIN_PCT", (float) voltagePercent);
        // Also try BAT1_V_EMPTY approach — force empty voltage
        commandSender.setParam("SIM_BAT_DRAIN", 1);
        System.out.println("SimulateBatteryFailsafeStep: Triggering battery failsafe at " + voltagePercent + "%");
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        // Complete once command is sent and brief delay passed
        if (getElapsedSeconds() >= 2.0) {
            markCompleted("battery failsafe triggered");
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return "[simulateBatteryFailsafe " + voltagePercent + "%]";
    }
}
