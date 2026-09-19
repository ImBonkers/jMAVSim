package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that injects a sensor failure by disabling a sensor via PX4 params.
 * Supported sensors: magnetometer, gyro, accel, baro, gps.
 */
public class InjectSensorFailureStep extends TestStep {
    private String sensor;

    public InjectSensorFailureStep(String sensor, double timeoutSeconds) {
        super("injectSensorFailure", timeoutSeconds);
        this.sensor = sensor.toLowerCase();
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        switch (sensor) {
            case "magnetometer":
            case "mag":
                commandSender.setParam("SIM_MAG1_FAIL", 1);
                break;
            case "gyro":
                commandSender.setParam("SIM_GYR1_FAIL", 1);
                break;
            case "accel":
            case "accelerometer":
                commandSender.setParam("SIM_ACCEL1_FAIL", 1);
                break;
            case "baro":
            case "barometer":
                commandSender.setParam("SIM_BARO1_FAIL", 1);
                break;
            case "gps":
                commandSender.setParam("SIM_GPS_BLOCK", 1);
                break;
            default:
                System.err.println("InjectSensorFailureStep: Unknown sensor: " + sensor);
                break;
        }
        System.out.println("InjectSensorFailureStep: Injected " + sensor + " failure");
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (getElapsedSeconds() >= 2.0) {
            markCompleted(sensor + " failure injected");
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return "[injectSensorFailure " + sensor + "]";
    }
}
