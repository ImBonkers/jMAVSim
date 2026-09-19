package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that commands the vehicle into an orbit pattern.
 * Uses OFFBOARD position setpoints to trace a circle if MAV_CMD_DO_ORBIT
 * is not supported. Completes after the specified number of laps.
 */
public class OrbitStep extends TestStep {
    private double radius;
    private double speed;
    private double altitude;
    private int laps;

    private double centerX;
    private double centerY;
    private double angle;
    private int lapCount;
    private double lastAngle;
    private boolean centerSet;
    private boolean offboardSet;
    private long lastCommandTime;

    public OrbitStep(double radius, double speed, double altitude, int laps, double timeoutSeconds) {
        super("orbit", timeoutSeconds);
        this.radius = radius;
        this.speed = speed;
        this.altitude = altitude;
        this.laps = laps;
        this.centerSet = false;
        this.offboardSet = false;
        this.lapCount = 0;
        this.angle = 0;
        this.lastAngle = 0;
        this.lastCommandTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        centerSet = false;
        offboardSet = false;
        lapCount = 0;
        angle = 0;
        lastAngle = 0;
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        if (!state.hasPosition) return;

        // Capture center on first valid position
        if (!centerSet) {
            centerX = state.x;
            centerY = state.y;
            centerSet = true;
            angle = 0;
        }

        // Advance angle based on speed and radius: angular_vel = speed / radius
        double dt = (currentTime - lastCommandTime) / 1000.0;
        if (dt <= 0) return;

        double angularVel = speed / radius;
        angle += angularVel * dt;

        // Count laps
        if (angle - lastAngle >= Math.PI * 2) {
            lapCount++;
            lastAngle += Math.PI * 2;
        }

        // Compute target position on circle
        double targetX = centerX + radius * Math.cos(angle);
        double targetY = centerY + radius * Math.sin(angle);
        double targetZ = -altitude;

        // Send at 10Hz
        commandSender.gotoPosition(targetX, targetY, targetZ);
        lastCommandTime = currentTime;

        if (!offboardSet && getElapsedSeconds() > 0.5) {
            commandSender.setOffboardMode();
            offboardSet = true;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (lapCount >= laps) {
            markCompleted(lapCount + " laps completed");
            return true;
        }
        return false;
    }

    @Override
    public String getProgressString(VehicleState state) {
        return String.format("[orbit] lap=%d/%d angle=%.0f°",
                lapCount, laps, Math.toDegrees(angle));
    }

    @Override
    public String getDisplayName() {
        return String.format("[orbit r=%.0fm %d laps]", radius, laps);
    }
}
