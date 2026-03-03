package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;
import me.drton.jmavsim.Environment;

import javax.vecmath.Vector3d;

/**
 * Step that sets environment wind conditions.
 * Completes immediately after setting wind.
 * This step manipulates the simulator environment directly.
 */
public class SetWindStep extends TestStep {
    private double windX;
    private double windY;
    private double windZ;
    private Environment environment;

    public SetWindStep(double x, double y, double z) {
        super("setWind", 1.0);  // Very short timeout since this completes immediately
        this.windX = x;
        this.windY = y;
        this.windZ = z;
        this.environment = null;
    }

    /**
     * Set the environment reference for wind manipulation.
     * Must be called before start().
     */
    public void setEnvironment(Environment env) {
        this.environment = env;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);

        if (environment != null) {
            Vector3d wind = new Vector3d(windX, windY, windZ);
            environment.setWind(wind);
            System.out.println(String.format("SetWindStep: Wind set to (%.1f, %.1f, %.1f) m/s", windX, windY, windZ));
        } else {
            System.err.println("SetWindStep: No environment reference set!");
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        // Wind setting is instantaneous
        markCompleted(null);
        return true;
    }

    @Override
    public String getDisplayName() {
        return String.format("[setWind %.1f,%.1f,%.1f]", windX, windY, windZ);
    }
}
