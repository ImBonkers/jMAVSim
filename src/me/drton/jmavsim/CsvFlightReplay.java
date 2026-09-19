package me.drton.jmavsim;

import me.drton.jmavlib.conversion.RotationConversion;
import me.drton.jmavsim.vehicle.AbstractVehicle;

import javax.vecmath.Matrix3d;
import javax.vecmath.Vector3d;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Replays a CSV flight log through the 3D visualizer.
 * Reads position and attitude from a CSV file (produced by FlightDataLogger)
 * and drives the vehicle model at real-time speed with interactive controls.
 */
public class CsvFlightReplay extends WorldObject {

    private final AbstractVehicle vehicle;
    private final List<ReplayFrame> frames;

    // Playback state
    private long playbackStartWallTime = -1;
    private long playbackOffsetMs = 0;
    private int currentFrameIndex = 0;
    private boolean replayPaused = true;  // start paused so user can orient
    private double speedMultiplier = 1.0;
    private boolean finished = false;

    private static class ReplayFrame {
        long elapsedMs;
        double x, y, z;
        double vx, vy, vz;
        double rollRad, pitchRad, yawRad;
    }

    public CsvFlightReplay(World world, AbstractVehicle vehicle, String csvPath) throws IOException {
        super(world);
        this.vehicle = vehicle;
        this.frames = new ArrayList<>();
        loadCsv(csvPath);
        if (frames.isEmpty()) {
            throw new IOException("CSV file contains no data rows: " + csvPath);
        }
        // Apply first frame immediately
        applyFrame(0);
    }

    private void loadCsv(String csvPath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(csvPath))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IOException("Empty CSV file");
            }

            // Resolve column indices from header
            String[] headers = headerLine.split(",");
            int colElapsed = indexOf(headers, "elapsed_ms");
            int colX = indexOf(headers, "x");
            int colY = indexOf(headers, "y");
            int colZ = indexOf(headers, "z");
            int colVx = indexOf(headers, "vx");
            int colVy = indexOf(headers, "vy");
            int colVz = indexOf(headers, "vz");
            int colRoll = indexOf(headers, "roll_deg");
            int colPitch = indexOf(headers, "pitch_deg");
            int colYaw = indexOf(headers, "yaw_deg");

            if (colElapsed < 0 || colX < 0 || colY < 0 || colZ < 0) {
                throw new IOException("CSV missing required columns (elapsed_ms, x, y, z)");
            }

            // Parse rows — deduplicate consecutive identical positions
            long lastElapsed = -1;
            String line;
            while ((line = reader.readLine()) != null) {
                String[] cols = line.split(",");
                if (cols.length <= colZ) continue;

                try {
                    long elapsed = Long.parseLong(cols[colElapsed].trim());
                    // Skip duplicate timestamps (CSV logs faster than MAVLink updates)
                    if (elapsed == lastElapsed) continue;
                    lastElapsed = elapsed;

                    ReplayFrame f = new ReplayFrame();
                    f.elapsedMs = elapsed;
                    f.x = parseDouble(cols, colX);
                    f.y = parseDouble(cols, colY);
                    f.z = parseDouble(cols, colZ);
                    f.vx = parseDouble(cols, colVx);
                    f.vy = parseDouble(cols, colVy);
                    f.vz = parseDouble(cols, colVz);
                    f.rollRad = colRoll >= 0 ? Math.toRadians(parseDouble(cols, colRoll)) : 0;
                    f.pitchRad = colPitch >= 0 ? Math.toRadians(parseDouble(cols, colPitch)) : 0;
                    f.yawRad = colYaw >= 0 ? Math.toRadians(parseDouble(cols, colYaw)) : 0;

                    // Skip NaN positions
                    if (Double.isNaN(f.x) || Double.isNaN(f.y) || Double.isNaN(f.z)) continue;

                    frames.add(f);
                } catch (NumberFormatException e) {
                    // Skip malformed rows
                }
            }
        }
    }

    private static int indexOf(String[] headers, String name) {
        for (int i = 0; i < headers.length; i++) {
            if (headers[i].trim().equals(name)) return i;
        }
        return -1;
    }

    private static double parseDouble(String[] cols, int index) {
        if (index < 0 || index >= cols.length) return 0;
        String val = cols[index].trim();
        if (val.isEmpty() || val.equalsIgnoreCase("nan")) return 0;
        return Double.parseDouble(val);
    }

    @Override
    public void update(long t, boolean paused) {
        if (replayPaused || finished || frames.size() < 2) return;

        // Initialize on first update
        if (playbackStartWallTime < 0) {
            playbackStartWallTime = System.currentTimeMillis();
            playbackOffsetMs = frames.get(currentFrameIndex).elapsedMs;
        }

        // Compute where we should be in the CSV timeline
        long wallElapsed = System.currentTimeMillis() - playbackStartWallTime;
        long targetCsvTime = playbackOffsetMs + (long)(wallElapsed * speedMultiplier);

        // Advance frame index
        while (currentFrameIndex < frames.size() - 2 &&
               frames.get(currentFrameIndex + 1).elapsedMs <= targetCsvTime) {
            currentFrameIndex++;
        }

        // Check end of data
        if (currentFrameIndex >= frames.size() - 1) {
            currentFrameIndex = frames.size() - 1;
            applyFrame(currentFrameIndex);
            finished = true;
            System.out.println("Replay finished. Press SPACE to restart.");
            return;
        }

        // Interpolate between current and next frame
        ReplayFrame f0 = frames.get(currentFrameIndex);
        ReplayFrame f1 = frames.get(currentFrameIndex + 1);
        long dt = f1.elapsedMs - f0.elapsedMs;
        double alpha = dt > 0 ? (double)(targetCsvTime - f0.elapsedMs) / dt : 0;
        alpha = Math.max(0, Math.min(1, alpha));

        double x = f0.x + alpha * (f1.x - f0.x);
        double y = f0.y + alpha * (f1.y - f0.y);
        double z = f0.z + alpha * (f1.z - f0.z);
        double vx = f0.vx + alpha * (f1.vx - f0.vx);
        double vy = f0.vy + alpha * (f1.vy - f0.vy);
        double vz = f0.vz + alpha * (f1.vz - f0.vz);
        double roll = lerpAngle(f0.rollRad, f1.rollRad, alpha);
        double pitch = lerpAngle(f0.pitchRad, f1.pitchRad, alpha);
        double yaw = lerpAngle(f0.yawRad, f1.yawRad, alpha);

        applyState(x, y, z, vx, vy, vz, roll, pitch, yaw);
    }

    private void applyFrame(int index) {
        ReplayFrame f = frames.get(index);
        applyState(f.x, f.y, f.z, f.vx, f.vy, f.vz, f.rollRad, f.pitchRad, f.yawRad);
    }

    private void applyState(double x, double y, double z,
                            double vx, double vy, double vz,
                            double roll, double pitch, double yaw) {
        vehicle.setPosition(new Vector3d(x, y, z));
        vehicle.setVelocity(new Vector3d(vx, vy, vz));

        double[] dcmArr = RotationConversion.rotationMatrixByEulerAngles(roll, pitch, yaw);
        Matrix3d dcm = new Matrix3d(dcmArr);
        vehicle.setRotation(dcm);
    }

    /** Linear interpolation for angles with shortest-path wrapping. */
    private static double lerpAngle(double a, double b, double t) {
        double diff = b - a;
        // Wrap to [-PI, PI]
        while (diff > Math.PI) diff -= 2 * Math.PI;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        return a + t * diff;
    }

    // --- Playback controls ---

    public void togglePause() {
        if (finished) {
            // Restart from beginning
            finished = false;
            currentFrameIndex = 0;
            replayPaused = false;
            playbackStartWallTime = System.currentTimeMillis();
            playbackOffsetMs = frames.get(0).elapsedMs;
            return;
        }

        if (replayPaused) {
            // Resuming — adjust wall time to maintain position
            playbackStartWallTime = System.currentTimeMillis() -
                (long)((frames.get(currentFrameIndex).elapsedMs - playbackOffsetMs) / speedMultiplier);
            replayPaused = false;
        } else {
            replayPaused = true;
        }
    }

    public void stepForward(int n) {
        int step = (int)(n * speedMultiplier);
        if (step < 1) step = 1;
        currentFrameIndex = Math.min(frames.size() - 1, currentFrameIndex + step);
        applyFrame(currentFrameIndex);
        finished = false;
    }

    public void stepBackward(int n) {
        int step = (int)(n * speedMultiplier);
        if (step < 1) step = 1;
        currentFrameIndex = Math.max(0, currentFrameIndex - step);
        applyFrame(currentFrameIndex);
        finished = false;
    }

    public void changeSpeed(double factor) {
        long currentCsvTime = frames.get(currentFrameIndex).elapsedMs;
        speedMultiplier *= factor;
        speedMultiplier = Math.max(0.125, Math.min(1024.0, speedMultiplier));
        // Reset wall time reference to maintain position
        if (!replayPaused) {
            playbackStartWallTime = System.currentTimeMillis() -
                (long)((currentCsvTime - playbackOffsetMs) / speedMultiplier);
        }
    }

    public void seekToStart() {
        currentFrameIndex = 0;
        finished = false;
        applyFrame(0);
        if (!replayPaused) {
            playbackStartWallTime = System.currentTimeMillis();
            playbackOffsetMs = frames.get(0).elapsedMs;
        }
    }

    // --- Getters ---

    public int getFrameCount() { return frames.size(); }
    public int getCurrentFrameIndex() { return currentFrameIndex; }
    public boolean isFinished() { return finished; }
    public boolean isPaused() { return replayPaused; }
    public double getSpeedMultiplier() { return speedMultiplier; }

    public long getDurationMs() {
        return frames.get(frames.size() - 1).elapsedMs - frames.get(0).elapsedMs;
    }

    public long getCurrentTimeMs() {
        return frames.get(currentFrameIndex).elapsedMs - frames.get(0).elapsedMs;
    }

    public String getStatusString() {
        long cur = getCurrentTimeMs();
        long total = getDurationMs();
        String timeStr = formatMs(cur) + " / " + formatMs(total);
        String speedStr = speedMultiplier == 1.0 ? "1x" :
            (speedMultiplier >= 1 ? String.format("%.0fx", speedMultiplier) :
                                    String.format("%.2fx", speedMultiplier));
        String stateStr = finished ? "END" : (replayPaused ? "PAUSED" : "PLAYING");
        return String.format("[%s] %s  %s  frame %d/%d",
            stateStr, timeStr, speedStr, currentFrameIndex + 1, frames.size());
    }

    private static String formatMs(long ms) {
        long secs = ms / 1000;
        if (secs >= 3600) {
            return String.format("%d:%02d:%02d", secs / 3600, (secs % 3600) / 60, secs % 60);
        }
        return String.format("%d:%02d", secs / 60, secs % 60);
    }
}
