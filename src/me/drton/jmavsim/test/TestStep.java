package me.drton.jmavsim.test;

/**
 * Abstract base class for test steps.
 * Each step has a type, timeout, and completion criteria.
 */
public abstract class TestStep {
    protected String type;
    protected double timeoutSeconds;
    protected long startTime;
    protected boolean started;
    protected boolean completed;
    protected boolean failed;
    protected String failureReason;
    protected String resultDetails;

    public TestStep(String type, double timeoutSeconds) {
        this.type = type;
        this.timeoutSeconds = timeoutSeconds;
        this.startTime = 0;
        this.started = false;
        this.completed = false;
        this.failed = false;
        this.failureReason = null;
        this.resultDetails = null;
    }

    /**
     * Get the step type name
     */
    public String getType() {
        return type;
    }

    /**
     * Get timeout in seconds
     */
    public double getTimeoutSeconds() {
        return timeoutSeconds;
    }

    /**
     * Check if step has started
     */
    public boolean isStarted() {
        return started;
    }

    /**
     * Check if step is completed (success or failure)
     */
    public boolean isCompleted() {
        return completed;
    }

    /**
     * Check if step failed
     */
    public boolean isFailed() {
        return failed;
    }

    /**
     * Get failure reason if failed
     */
    public String getFailureReason() {
        return failureReason;
    }

    /**
     * Get result details string
     */
    public String getResultDetails() {
        return resultDetails;
    }

    /**
     * Get elapsed time since start in seconds
     */
    public double getElapsedSeconds() {
        if (!started) return 0;
        return (System.currentTimeMillis() - startTime) / 1000.0;
    }

    /**
     * Start the step. Called once when the step begins.
     * Override to send commands.
     */
    public void start(CommandSender commandSender, long currentTime) {
        this.startTime = currentTime;
        this.started = true;
    }

    /**
     * Called every update cycle while step is active.
     * Use to resend commands if needed.
     */
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        // Default: do nothing
    }

    /**
     * Check if the step completion criteria are met.
     * @param state Current vehicle state
     * @return true if step is complete
     */
    public abstract boolean checkComplete(VehicleState state);

    /**
     * Check for timeout
     * @param currentTime Current time in ms
     * @return true if timed out
     */
    public boolean checkTimeout(long currentTime) {
        if (!started) return false;
        double elapsed = (currentTime - startTime) / 1000.0;
        return elapsed > timeoutSeconds;
    }

    /**
     * Mark step as completed successfully
     */
    protected void markCompleted(String details) {
        this.completed = true;
        this.failed = false;
        this.resultDetails = details;
    }

    /**
     * Mark step as failed
     */
    public void markFailed(String reason) {
        this.completed = true;
        this.failed = true;
        this.failureReason = reason;
    }

    /**
     * Get display name for progress output
     */
    public String getDisplayName() {
        return "[" + type + "]";
    }

    @Override
    public String toString() {
        return type + "(timeout=" + timeoutSeconds + "s)";
    }
}
