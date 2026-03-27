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
 *
 * Supports two modes:
 *   1. Static wind: set a fixed wind vector (x, y, z)
 *   2. Fluctuating wind: set a direction vector + min/max strength range.
 *      The wind will fluctuate in real time between minStrength and maxStrength
 *      along the given direction, using the simulator's built-in wind deviation model.
 */
public class SetWindStep extends TestStep {
    private double windX;
    private double windY;
    private double windZ;

    // Fluctuation range (null means static wind)
    private Double minStrength;
    private Double maxStrength;

    private Environment environment;

    /**
     * Create a static wind step (no fluctuation).
     */
    public SetWindStep(double x, double y, double z) {
        super("setWind", 1.0);  // Very short timeout since this completes immediately
        this.windX = x;
        this.windY = y;
        this.windZ = z;
        this.minStrength = null;
        this.maxStrength = null;
        this.environment = null;
        this.critical = false;  // Wind setting is non-critical by default
    }

    /**
     * Create a fluctuating wind step.
     * Wind blows along the direction (x, y, z) with strength varying between
     * minStrength and maxStrength m/s in real time.
     *
     * @param x Direction X component (will be normalized)
     * @param y Direction Y component (will be normalized)
     * @param z Direction Z component (will be normalized)
     * @param minStrength Minimum wind speed in m/s
     * @param maxStrength Maximum wind speed in m/s
     */
    public SetWindStep(double x, double y, double z, double minStrength, double maxStrength) {
        super("setWind", 1.0);
        this.windX = x;
        this.windY = y;
        this.windZ = z;
        this.minStrength = minStrength;
        this.maxStrength = maxStrength;
        this.environment = null;
        this.critical = false;
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

        if (environment == null) {
            System.err.println("SetWindStep: No environment reference set!");
            return;
        }

        if (minStrength != null && maxStrength != null) {
            applyFluctuatingWind();
        } else {
            applyStaticWind();
        }
    }

    private void applyStaticWind() {
        Vector3d wind = new Vector3d(windX, windY, windZ);
        environment.setWind(wind);
        // Zero out deviation for static wind
        environment.setWindDeviation(new Vector3d(0, 0, 0));
        System.out.println(String.format(
                "SetWindStep: Static wind set to (%.1f, %.1f, %.1f) m/s",
                windX, windY, windZ));
    }

    private void applyFluctuatingWind() {
        // Normalize direction vector
        double length = Math.sqrt(windX * windX + windY * windY + windZ * windZ);
        double dirX, dirY, dirZ;
        if (length < 1e-9) {
            // Zero direction defaults to North
            dirX = 1.0;
            dirY = 0.0;
            dirZ = 0.0;
        } else {
            dirX = windX / length;
            dirY = windY / length;
            dirZ = windZ / length;
        }

        // Base wind = direction * midpoint of range
        double midStrength = (minStrength + maxStrength) / 2.0;
        Vector3d baseWind = new Vector3d(
                dirX * midStrength,
                dirY * midStrength,
                dirZ * midStrength);

        // Deviation controls the Gaussian noise amplitude per axis.
        // The existing model in SimpleEnvironment.update() applies:
        //   noise = windDeviation * gaussian()  (per-axis, scaled by dt)
        //   windCurrent drifts toward baseWind with time constant windT
        //
        // To make ~95% of samples fall within [min, max], set deviation
        // so that 2*sigma covers the half-range. The dt scaling and mean-reversion
        // in the model naturally bound the actual fluctuation.
        double halfRange = (maxStrength - minStrength) / 2.0;
        double deviation = halfRange / 2.0;  // ~95% within range
        Vector3d windDev = new Vector3d(
                Math.abs(dirX) * deviation,
                Math.abs(dirY) * deviation,
                Math.abs(dirZ) * deviation);

        environment.setWind(baseWind);
        environment.setWindDeviation(windDev);

        // Seed current wind at the midpoint so it starts within range
        environment.setCurrentWind(new Vector3d(baseWind));

        System.out.println(String.format(
                "SetWindStep: Fluctuating wind dir=(%.2f,%.2f,%.2f) range=[%.1f, %.1f] m/s " +
                "(base=%.1f, dev=%.1f)",
                dirX, dirY, dirZ, minStrength, maxStrength, midStrength, deviation));
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        // Wind setting is instantaneous
        markCompleted(null);
        return true;
    }

    @Override
    public String getDisplayName() {
        if (minStrength != null && maxStrength != null) {
            return String.format("[setWind dir=(%.1f,%.1f,%.1f) %.1f-%.1f m/s]",
                    windX, windY, windZ, minStrength, maxStrength);
        }
        return String.format("[setWind %.1f,%.1f,%.1f]", windX, windY, windZ);
    }
}
