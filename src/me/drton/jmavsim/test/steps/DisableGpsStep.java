package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that disables GPS for a specified duration, then re-enables it.
 * Tries multiple PX4 param names to cover SITL and HITL configurations.
 * Completion is purely time-based — does not depend on param ACK.
 */
public class DisableGpsStep extends TestStep {
    private double disableDurationSeconds;
    private boolean restored;
    private long disableTime;

    public DisableGpsStep(double disableDurationSeconds, double timeoutSeconds) {
        // Timeout must exceed disable duration + recovery margin
        super("disableGps", Math.max(timeoutSeconds, disableDurationSeconds + 5.0));
        this.disableDurationSeconds = disableDurationSeconds;
        this.restored = false;
        this.disableTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        // Try multiple param names to cover different PX4 versions/configs
        commandSender.setParam("SIM_GPS_BLOCK", 1);
        commandSender.setParam("SIM_GPS1_BLOCK", 1);
        restored = false;
        disableTime = currentTime;
        System.out.println("DisableGpsStep: GPS disabled for " + disableDurationSeconds + "s");
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        if (!restored) {
            double elapsed = (currentTime - disableTime) / 1000.0;
            if (elapsed >= disableDurationSeconds) {
                commandSender.setParam("SIM_GPS_BLOCK", 0);
                commandSender.setParam("SIM_GPS1_BLOCK", 0);
                restored = true;
                System.out.println("DisableGpsStep: GPS restored");
            }
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        // Complete once GPS has been restored and 2s recovery time has passed
        if (restored && getElapsedSeconds() >= disableDurationSeconds + 2.0) {
            markCompleted("disabled " + disableDurationSeconds + "s");
            return true;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        String gpsState = restored ? "restored" : "disabled";
        return String.format("[disableGps] state=%s elapsed=%.1fs/%.0fs",
                gpsState, getElapsedSeconds(), disableDurationSeconds);
    }

    @Override
    public String getDisplayName() {
        return String.format("[disableGps %.0fs]", disableDurationSeconds);
    }
}
