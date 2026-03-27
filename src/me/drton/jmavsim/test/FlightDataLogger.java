package me.drton.jmavsim.test;

import me.drton.jmavsim.Environment;

import javax.vecmath.Vector3d;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Logs time-series flight telemetry and FC diagnostics to a CSV file during
 * scenario execution. Each row is a snapshot recorded at a fixed interval
 * from the simulation loop.
 *
 * Columns cover:
 *   - Timing: elapsed ms, wall clock, FC boot time
 *   - Scenario: step index, step type
 *   - Vehicle state: position, velocity, attitude, altitude, speed, armed/mode
 *   - FC health (SYS_STATUS): CPU load, comm drop rate, comm errors, error counters
 *   - EKF (ESTIMATOR_STATUS): flags, innovation ratios, position accuracy
 *   - Vibration: per-axis vibration levels, accelerometer clipping counts
 *   - Environment: current wind vector
 */
public class FlightDataLogger {
    private PrintWriter writer;
    private long scenarioStartTime;
    private boolean open;
    private String filePath;

    private static final String HEADER =
            // Timing
            "elapsed_ms,wall_clock,boot_time_ms," +
            // Scenario context
            "step_index,step_type," +
            // Position & velocity (NED)
            "x,y,z,vx,vy,vz," +
            // Attitude
            "roll_deg,pitch_deg,yaw_deg," +
            // Derived
            "altitude,ground_speed,vertical_speed," +
            // Mode
            "armed,base_mode,custom_mode," +
            // FC health (SYS_STATUS)
            "cpu_load,comm_drop_rate,comm_errors," +
            "errors_count1,errors_count2,errors_count3,errors_count4," +
            // EKF (ESTIMATOR_STATUS)
            "ekf_flags,ekf_vel_ratio,ekf_pos_horiz_ratio,ekf_pos_vert_ratio," +
            "ekf_mag_ratio,ekf_pos_horiz_accuracy,ekf_pos_vert_accuracy," +
            // Vibration
            "vibration_x,vibration_y,vibration_z," +
            "clipping_0,clipping_1,clipping_2," +
            // Attitude setpoint & error (ATTITUDE_TARGET)
            "roll_sp_deg,pitch_sp_deg,yaw_sp_deg,thrust_sp," +
            "roll_err_deg,pitch_err_deg,yaw_err_deg," +
            // NPU stats (NAMED_VALUE_FLOAT)
            "npu_ms,npu_fps,npu_avg," +
            // Environment
            "wind_x,wind_y,wind_z";

    /**
     * Create a flight data logger.
     * @param outputDir Directory to write the CSV file
     * @param scenarioName Scenario name for the filename
     */
    public FlightDataLogger(String outputDir, String scenarioName) {
        this.open = false;
        this.scenarioStartTime = 0;

        if (outputDir == null || outputDir.isEmpty()) {
            this.filePath = null;
            return;
        }

        File dir = new File(outputDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String fileName = scenarioName + "_flight_" + sdf.format(new Date()) + ".csv";
        this.filePath = new File(dir, fileName).getAbsolutePath();
    }

    /**
     * Start logging. Call once when the scenario begins running.
     * @param startTime Scenario start time in ms (System.currentTimeMillis)
     */
    public void start(long startTime) {
        this.scenarioStartTime = startTime;

        if (filePath == null) {
            return;
        }

        try {
            writer = new PrintWriter(new BufferedWriter(new FileWriter(filePath)));
            writer.println(HEADER);
            open = true;
            System.out.println("Flight data logging to: " + filePath);
        } catch (IOException e) {
            System.err.println("Failed to open flight data log: " + e.getMessage());
            open = false;
        }
    }

    /**
     * Record one telemetry sample.
     * @param state Current vehicle state snapshot
     * @param environment Environment for wind data (may be null)
     * @param stepIndex Current step index (-1 if none)
     * @param stepType Current step type name (null if none)
     * @param currentTime Current wall-clock time in ms
     */
    public void record(VehicleState state, Environment environment,
                       int stepIndex, String stepType, long currentTime) {
        if (!open || writer == null) {
            return;
        }

        if (!state.hasPosition && !state.hasHeartbeat) {
            return;  // No useful data yet
        }

        long elapsed = currentTime - scenarioStartTime;

        // Get current wind from environment
        double wx = 0, wy = 0, wz = 0;
        if (environment != null) {
            Vector3d wind = environment.getCurrentWind(null);
            if (wind != null) {
                wx = wind.x;
                wy = wind.y;
                wz = wind.z;
            }
        }

        writer.printf(
                // Timing
                "%d,%d,%d," +
                // Scenario
                "%d,%s," +
                // Position & velocity
                "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f," +
                // Attitude
                "%.2f,%.2f,%.2f," +
                // Derived
                "%.4f,%.4f,%.4f," +
                // Mode
                "%s,%d,%d," +
                // FC health
                "%d,%d,%d,%d,%d,%d,%d," +
                // EKF
                "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f," +
                // Vibration
                "%.6f,%.6f,%.6f,%d,%d,%d," +
                // Attitude setpoint & error
                "%.2f,%.2f,%.2f,%.4f,%.2f,%.2f,%.2f," +
                // NPU
                "%.3f,%.2f,%.3f," +
                // Wind
                "%.4f,%.4f,%.4f%n",
                // --- values ---
                elapsed, currentTime, state.bootTimeMs,
                stepIndex, stepType != null ? stepType : "",
                state.x, state.y, state.z, state.vx, state.vy, state.vz,
                Math.toDegrees(state.roll), Math.toDegrees(state.pitch), Math.toDegrees(state.yaw),
                state.getAltitude(), state.getGroundSpeed(), state.getVerticalSpeed(),
                state.armed, state.baseMode, state.customMode,
                state.cpuLoad, state.commDropRate, state.commErrors,
                state.errorsCount1, state.errorsCount2, state.errorsCount3, state.errorsCount4,
                state.ekfFlags, state.ekfVelRatio, state.ekfPosHorizRatio, state.ekfPosVertRatio,
                state.ekfMagRatio, state.ekfPosHorizAccuracy, state.ekfPosVertAccuracy,
                state.vibrationX, state.vibrationY, state.vibrationZ,
                state.clipping0, state.clipping1, state.clipping2,
                Math.toDegrees(state.rollSetpoint), Math.toDegrees(state.pitchSetpoint), Math.toDegrees(state.yawSetpoint),
                state.thrustSetpoint,
                state.hasAttitudeTarget ? Math.toDegrees(state.roll - state.rollSetpoint) : 0.0,
                state.hasAttitudeTarget ? Math.toDegrees(state.pitch - state.pitchSetpoint) : 0.0,
                state.hasAttitudeTarget ? Math.toDegrees(state.yaw - state.yawSetpoint) : 0.0,
                state.npuMs, state.npuFps, state.npuAvg,
                wx, wy, wz);
    }

    /**
     * Stop logging and close the file.
     */
    public void stop() {
        if (open && writer != null) {
            writer.flush();
            writer.close();
            open = false;
            System.out.println("Flight data log closed: " + filePath);
        }
    }

    /**
     * Check if the logger is actively recording.
     */
    public boolean isOpen() {
        return open;
    }

    /**
     * Get the output file path, or null if logging is disabled.
     */
    public String getFilePath() {
        return filePath;
    }
}
