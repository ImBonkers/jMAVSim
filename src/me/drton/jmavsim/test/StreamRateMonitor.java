package me.drton.jmavsim.test;

import java.util.HashMap;
import java.util.Map;

/**
 * Tracks MAVLink message arrival rates and requests corrections when
 * the observed rate is too far from the desired rate.
 *
 * Used by TestScenarioRunner to ensure both the STM32N6 and reference
 * boards stream telemetry at identical rates for fair comparison.
 */
public class StreamRateMonitor {

    /** Desired rate config: msgId -> intervalUs */
    private final Map<Integer, Long> desiredRates = new HashMap<>();

    /** Tracking state per message ID */
    private final Map<Integer, RateTracker> trackers = new HashMap<>();

    /** How far off the rate can be before we re-request (0.3 = 30%) */
    private static final double TOLERANCE = 0.3;

    /** Minimum samples before we evaluate the rate */
    private static final int MIN_SAMPLES = 5;

    /** Don't re-request more often than this (ms) */
    private static final long REQUEST_COOLDOWN_MS = 5000;

    // MAVLink message IDs
    public static final int MSG_ATTITUDE = 30;
    public static final int MSG_ATTITUDE_TARGET = 83;
    public static final int MSG_LOCAL_POSITION_NED = 32;
    public static final int MSG_SYS_STATUS = 1;
    public static final int MSG_ESTIMATOR_STATUS = 230;
    public static final int MSG_HIGHRES_IMU = 105;
    public static final int MSG_VIBRATION = 241;
    public static final int MSG_HIL_ACTUATOR_CONTROLS = 93;

    /** Observe-only streams: track rate + jitter but don't request */
    private final Map<Integer, RateTracker> observeOnly = new HashMap<>();

    public StreamRateMonitor() {
        // Default desired rates for thesis data capture
        setDesiredRate(MSG_ATTITUDE, 50);
        setDesiredRate(MSG_ATTITUDE_TARGET, 50);
        setDesiredRate(MSG_LOCAL_POSITION_NED, 50);
        setDesiredRate(MSG_SYS_STATUS, 2);
        setDesiredRate(MSG_ESTIMATOR_STATUS, 5);
        setDesiredRate(MSG_HIGHRES_IMU, 50);
        setDesiredRate(MSG_VIBRATION, 2);

        // Observe control loop rate (driven by FC, not requestable)
        trackOnly(MSG_HIL_ACTUATOR_CONTROLS);
    }

    /**
     * Track message rate without requesting — for FC-driven streams.
     */
    public void trackOnly(int msgId) {
        observeOnly.put(msgId, new RateTracker());
    }

    /**
     * Set desired rate for a message ID.
     * @param msgId MAVLink message ID
     * @param rateHz desired rate in Hz
     */
    public void setDesiredRate(int msgId, double rateHz) {
        long intervalUs = rateHz > 0 ? (long)(1e6 / rateHz) : 0;
        desiredRates.put(msgId, intervalUs);
        trackers.put(msgId, new RateTracker());
    }

    /**
     * Called when a message is received. Tracks arrival time.
     * @param msgId MAVLink message ID
     */
    public void messageReceived(int msgId) {
        RateTracker tracker = trackers.get(msgId);
        if (tracker != null) {
            tracker.recordArrival();
        }

        RateTracker obs = observeOnly.get(msgId);
        if (obs != null) {
            obs.recordArrival();
        }
    }

    /**
     * Check all tracked streams and return requests needed.
     * @return map of msgId -> intervalUs for streams that need correction
     */
    public Map<Integer, Long> getNeededRequests() {
        Map<Integer, Long> requests = new HashMap<>();
        long now = System.currentTimeMillis();

        for (Map.Entry<Integer, Long> entry : desiredRates.entrySet()) {
            int msgId = entry.getKey();
            long desiredIntervalUs = entry.getValue();
            if (desiredIntervalUs <= 0) continue;

            RateTracker tracker = trackers.get(msgId);
            if (tracker == null) continue;

            // Cooldown check
            if (now - tracker.lastRequestTime < REQUEST_COOLDOWN_MS) continue;

            double desiredHz = 1e6 / desiredIntervalUs;

            if (tracker.count < MIN_SAMPLES) {
                // Not enough data yet — request if we've been waiting > 2s
                if (now - tracker.firstSeen > 2000 || tracker.count == 0) {
                    requests.put(msgId, desiredIntervalUs);
                    tracker.lastRequestTime = now;
                }
            } else {
                double observedHz = tracker.getRate();
                double error = Math.abs(observedHz - desiredHz) / desiredHz;

                if (error > TOLERANCE) {
                    requests.put(msgId, desiredIntervalUs);
                    tracker.lastRequestTime = now;
                }
            }
        }

        return requests;
    }

    /**
     * Get the observed rate for a message ID.
     * @return rate in Hz, or 0 if not enough data
     */
    public double getObservedRate(int msgId) {
        RateTracker tracker = trackers.get(msgId);
        if (tracker == null) tracker = observeOnly.get(msgId);
        return (tracker != null && tracker.count >= MIN_SAMPLES) ? tracker.getRate() : 0.0;
    }

    /**
     * Get jitter (stddev of inter-arrival times) for a tracked message.
     */
    public double getJitterMs(int msgId) {
        RateTracker tracker = trackers.get(msgId);
        if (tracker == null) tracker = observeOnly.get(msgId);
        return (tracker != null) ? tracker.getJitterMs() : 0.0;
    }

    /**
     * Get maximum inter-arrival gap for a tracked message.
     */
    public double getMaxGapMs(int msgId) {
        RateTracker tracker = trackers.get(msgId);
        if (tracker == null) tracker = observeOnly.get(msgId);
        return (tracker != null) ? tracker.maxGapMs : 0.0;
    }

    /**
     * Print current rates vs desired for debugging.
     */
    public String getRateSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Stream rates:\n");
        for (Map.Entry<Integer, Long> entry : desiredRates.entrySet()) {
            int msgId = entry.getKey();
            double desiredHz = 1e6 / entry.getValue();
            double observedHz = getObservedRate(msgId);
            String name = getMessageName(msgId);
            sb.append(String.format("  %-25s desired=%5.0f Hz, observed=%6.1f Hz, jitter=%5.1f ms%s\n",
                name, desiredHz, observedHz, getJitterMs(msgId),
                observedHz > 0 && Math.abs(observedHz - desiredHz) / desiredHz > TOLERANCE ? " [MISMATCH]" : ""));
        }
        sb.append("Control loop:\n");
        for (Map.Entry<Integer, RateTracker> entry : observeOnly.entrySet()) {
            int msgId = entry.getKey();
            RateTracker t = entry.getValue();
            String name = getMessageName(msgId);
            sb.append(String.format("  %-25s rate=%.1f Hz, jitter=%.2f ms, max_gap=%.1f ms, n=%d\n",
                name, t.getRate(), t.getJitterMs(), t.maxGapMs, t.count));
        }
        return sb.toString();
    }

    private String getMessageName(int msgId) {
        switch (msgId) {
            case MSG_ATTITUDE: return "ATTITUDE (30)";
            case MSG_ATTITUDE_TARGET: return "ATTITUDE_TARGET (83)";
            case MSG_LOCAL_POSITION_NED: return "LOCAL_POSITION_NED (32)";
            case MSG_SYS_STATUS: return "SYS_STATUS (1)";
            case MSG_ESTIMATOR_STATUS: return "ESTIMATOR_STATUS (230)";
            case MSG_HIGHRES_IMU: return "HIGHRES_IMU (105)";
            case MSG_VIBRATION: return "VIBRATION (241)";
            case MSG_HIL_ACTUATOR_CONTROLS: return "HIL_ACTUATOR_CTRL (93)";
            default: return "MSG_" + msgId;
        }
    }

    /** Tracks arrival times for a single message type */
    private static class RateTracker {
        long firstSeen;
        long lastSeen;
        int count;
        long lastRequestTime;
        double maxGapMs;
        // Running variance (Welford's algorithm)
        private double m2;
        private double mean;
        private int jitterCount;

        void recordArrival() {
            long now = System.currentTimeMillis();
            if (count == 0) {
                firstSeen = now;
            } else {
                double gap = now - lastSeen;
                if (gap > maxGapMs) maxGapMs = gap;
                // Update running variance
                jitterCount++;
                double delta = gap - mean;
                mean += delta / jitterCount;
                double delta2 = gap - mean;
                m2 += delta * delta2;
            }
            lastSeen = now;
            count++;
        }

        double getRate() {
            if (count < 2) return 0;
            double elapsed = (lastSeen - firstSeen) / 1000.0;
            return elapsed > 0 ? (count - 1) / elapsed : 0;
        }

        double getJitterMs() {
            if (jitterCount < 2) return 0;
            return Math.sqrt(m2 / (jitterCount - 1));
        }
    }
}
