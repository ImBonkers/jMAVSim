package me.drton.jmavsim.test;

import me.drton.jmavsim.Environment;
import me.drton.jmavsim.World;
import me.drton.jmavsim.WorldObject;
import me.drton.jmavsim.test.steps.SetWindStep;
import me.drton.jmavsim.test.steps.RebootStep;

/**
 * Main orchestrator for running test scenarios.
 * Implements WorldObject so update() is called at simulation rate.
 */
public class TestScenarioRunner extends WorldObject {
    private enum State {
        WAITING_FOR_CONNECTION,
        RUNNING,
        COMPLETED,
        FAILED
    }

    private TestScenario scenario;
    private StateMonitor stateMonitor;
    private CommandSender commandSender;
    private TestReporter reporter;
    private FlightDataLogger dataLogger;
    private Environment environment;
    private StreamRateMonitor rateMonitor;

    private State state;
    private int currentStepIndex;
    private long scenarioStartTime;
    private long lastProgressTime;
    private long lastLogTime;
    private long lastStreamRequestTime;
    private int exitCode;

    private static final long PROGRESS_INTERVAL_MS = 500;
    private static final long LOG_INTERVAL_MS = 4;  // 250 Hz telemetry logging
    private static final long STREAM_REQUEST_INTERVAL_MS = 1000;  // Re-request missing streams every 1s
    private static final long CONNECTION_TIMEOUT_MS = 30000;

    private Runnable onCompleteCallback;

    public TestScenarioRunner(World world, TestScenario scenario, StateMonitor stateMonitor,
                              CommandSender commandSender, String outputDir) {
        super(world);
        this.scenario = scenario;
        this.stateMonitor = stateMonitor;
        this.commandSender = commandSender;
        this.reporter = new TestReporter(scenario, outputDir);
        this.dataLogger = new FlightDataLogger(outputDir, scenario.getName());
        this.environment = world.getEnvironment();

        this.rateMonitor = new StreamRateMonitor();
        stateMonitor.setRateMonitor(rateMonitor);

        this.state = State.WAITING_FOR_CONNECTION;
        this.currentStepIndex = -1;
        this.scenarioStartTime = 0;
        this.lastProgressTime = 0;
        this.lastLogTime = 0;
        this.lastStreamRequestTime = 0;
        this.exitCode = 0;
        this.onCompleteCallback = null;
    }

    /**
     * Set callback to be invoked when scenario completes
     */
    public void setOnComplete(Runnable callback) {
        this.onCompleteCallback = callback;
    }

    /**
     * Get exit code (0 = success, 1 = failure)
     */
    public int getExitCode() {
        return exitCode;
    }

    /**
     * Check if scenario has completed
     */
    public boolean isCompleted() {
        return state == State.COMPLETED || state == State.FAILED;
    }

    @Override
    public void update(long t, boolean paused) {
        if (paused) return;

        long currentTime = System.currentTimeMillis();

        switch (state) {
            case WAITING_FOR_CONNECTION:
                handleWaitingForConnection(currentTime);
                break;

            case RUNNING:
                handleRunning(currentTime);
                break;

            case COMPLETED:
            case FAILED:
                // Nothing to do
                break;
        }
    }

    private void handleWaitingForConnection(long currentTime) {
        if (scenarioStartTime == 0) {
            scenarioStartTime = currentTime;
            System.out.println("Waiting for vehicle connection...");
        }

        if (stateMonitor.isConnected()) {
            // Connected - start scenario
            System.out.println("Vehicle connected (sysId=" + stateMonitor.getTargetSysId() + ")");

            // Sync target system ID with command sender
            if (stateMonitor.getTargetSysId() > 0) {
                commandSender.setTargetSysId(stateMonitor.getTargetSysId());
            }

            state = State.RUNNING;
            scenarioStartTime = currentTime;
            currentStepIndex = -1;
            lastLogTime = currentTime;

            reporter.printHeader();
            dataLogger.start(scenarioStartTime);
            advanceToNextStep(currentTime);

        } else if (currentTime - scenarioStartTime > CONNECTION_TIMEOUT_MS) {
            System.err.println("ERROR: Connection timeout - no heartbeat received");
            state = State.FAILED;
            exitCode = 1;
            invokeCallback();
        }
    }

    private void handleRunning(long currentTime) {
        if (currentStepIndex < 0 || currentStepIndex >= scenario.getStepCount()) {
            // All steps completed
            finishScenario();
            return;
        }

        TestStep currentStep = scenario.getSteps().get(currentStepIndex);
        VehicleState vehicleState = stateMonitor.getState();

        // Log flight data at fixed interval
        if (currentTime - lastLogTime >= LOG_INTERVAL_MS) {
            String stepType = currentStep.isStarted() ? currentStep.getType() : null;
            dataLogger.record(vehicleState, environment, currentStepIndex, stepType, currentTime);
            lastLogTime = currentTime;
        }

        // Check stream rates and request corrections
        if (currentTime - lastStreamRequestTime >= STREAM_REQUEST_INTERVAL_MS) {
            requestStreamRateCorrections();
            lastStreamRequestTime = currentTime;
        }

        // Check global timeout
        double scenarioElapsed = (currentTime - scenarioStartTime) / 1000.0;
        if (scenarioElapsed > scenario.getGlobalTimeoutSeconds()) {
            currentStep.markFailed("Global timeout exceeded");
            reporter.printStepResult(currentStep, currentStepIndex);
            System.err.println("ERROR: Global scenario timeout exceeded");
            state = State.FAILED;
            exitCode = 1;
            reporter.printSummary();
            dataLogger.stop();
            invokeCallback();
            return;
        }

        // Start step if not started
        if (!currentStep.isStarted()) {
            reporter.printStepStart(currentStep, currentStepIndex);
            currentStep.start(commandSender, currentTime);
            lastProgressTime = currentTime;
        }

        // Update step (allows resending commands, etc.)
        currentStep.update(commandSender, vehicleState, currentTime);

        // Check completion
        if (currentStep.checkComplete(vehicleState)) {
            // Verify the step properly marked itself as completed or failed
            if (!currentStep.isCompleted()) {
                System.err.println("WARNING: Step [" + currentStep.getType() +
                        "] returned true from checkComplete() without calling " +
                        "markCompleted() or markFailed() — forcing failure");
                currentStep.markFailed("Step did not mark completion state");
            }
            reporter.printStepResult(currentStep, currentStepIndex);
            // If the step marked itself failed during checkComplete, handle it
            if (currentStep.isFailed() && currentStep.isCritical()) {
                System.err.println("ERROR: Critical step [" + currentStep.getType() +
                        "] failed — aborting scenario");
                state = State.FAILED;
                exitCode = 1;
                reporter.printSummary();
                dataLogger.stop();
                invokeCallback();
                return;
            }
            advanceToNextStep(currentTime);
            return;
        }

        // Check timeout
        if (currentStep.checkTimeout(currentTime)) {
            currentStep.markFailed("Timeout — condition never met");
            reporter.printStepResult(currentStep, currentStepIndex);
            if (currentStep.isCritical()) {
                System.err.println("ERROR: Critical step [" + currentStep.getType() +
                        "] failed — aborting scenario");
                state = State.FAILED;
                exitCode = 1;
                reporter.printSummary();
                dataLogger.stop();
                invokeCallback();
                return;
            }
            advanceToNextStep(currentTime);
            return;
        }

        // Print progress
        if (currentTime - lastProgressTime > PROGRESS_INTERVAL_MS) {
            String progress = currentStep.getProgressString(vehicleState);
            if (progress != null) {
                reporter.printProgressString(progress);
            } else {
                reporter.printProgress();
            }
            lastProgressTime = currentTime;
        }
    }

    private void advanceToNextStep(long currentTime) {
        // Verify current step is in a terminal state before advancing
        if (currentStepIndex >= 0 && currentStepIndex < scenario.getStepCount()) {
            TestStep prevStep = scenario.getSteps().get(currentStepIndex);
            if (!prevStep.isCompleted()) {
                System.err.println("BUG: Attempting to advance past step [" +
                        prevStep.getType() + "] which is not completed or failed — blocking");
                return;
            }

        }

        currentStepIndex++;

        if (currentStepIndex >= scenario.getStepCount()) {
            // All steps done
            finishScenario();
            return;
        }

        // Prepare next step
        TestStep nextStep = scenario.getSteps().get(currentStepIndex);

        // Special handling for SetWindStep - inject environment reference
        if (nextStep instanceof SetWindStep && environment != null) {
            ((SetWindStep) nextStep).setEnvironment(environment);
        }

        // Special handling for RebootStep - inject state monitor reference
        if (nextStep instanceof RebootStep) {
            ((RebootStep) nextStep).setStateMonitor(stateMonitor);
        }
    }

    /**
     * Check all stream rates and request corrections for any that are
     * missing or too far from the desired rate.
     */
    private void requestStreamRateCorrections() {
        java.util.Map<Integer, Long> needed = rateMonitor.getNeededRequests();
        for (java.util.Map.Entry<Integer, Long> entry : needed.entrySet()) {
            commandSender.requestMessageInterval(entry.getKey(), entry.getValue());
        }
    }

    private void finishScenario() {
        dataLogger.stop();
        System.out.println(rateMonitor.getRateSummary());
        reporter.printSummary();
        exitCode = reporter.getExitCode();
        state = exitCode == 0 ? State.COMPLETED : State.FAILED;
        invokeCallback();
    }

    private void invokeCallback() {
        if (onCompleteCallback != null) {
            onCompleteCallback.run();
        }
    }
}
