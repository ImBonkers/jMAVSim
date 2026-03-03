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
    private Environment environment;

    private State state;
    private int currentStepIndex;
    private long scenarioStartTime;
    private long lastProgressTime;
    private int exitCode;

    private static final long PROGRESS_INTERVAL_MS = 500;
    private static final long CONNECTION_TIMEOUT_MS = 30000;

    private Runnable onCompleteCallback;

    public TestScenarioRunner(World world, TestScenario scenario, StateMonitor stateMonitor,
                              CommandSender commandSender, String outputDir) {
        super(world);
        this.scenario = scenario;
        this.stateMonitor = stateMonitor;
        this.commandSender = commandSender;
        this.reporter = new TestReporter(scenario, outputDir);
        this.environment = world.getEnvironment();

        this.state = State.WAITING_FOR_CONNECTION;
        this.currentStepIndex = -1;
        this.scenarioStartTime = 0;
        this.lastProgressTime = 0;
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

            reporter.printHeader();
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

        // Check global timeout
        double scenarioElapsed = (currentTime - scenarioStartTime) / 1000.0;
        if (scenarioElapsed > scenario.getGlobalTimeoutSeconds()) {
            currentStep.markFailed("Global timeout exceeded");
            reporter.printStepResult(currentStep, currentStepIndex);
            System.err.println("ERROR: Global scenario timeout exceeded");
            state = State.FAILED;
            exitCode = 1;
            reporter.printSummary();
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
            // Step completed successfully
            if (currentStep instanceof SetWindStep) {
                // SetWind is instantaneous - special handling
            }
            reporter.printStepResult(currentStep, currentStepIndex);
            advanceToNextStep(currentTime);
            return;
        }

        // Check timeout
        if (currentStep.checkTimeout(currentTime)) {
            currentStep.markFailed("Timeout");
            reporter.printStepResult(currentStep, currentStepIndex);
            // Continue to next step or fail scenario based on step type
            advanceToNextStep(currentTime);
            return;
        }

        // Print progress dots
        if (currentTime - lastProgressTime > PROGRESS_INTERVAL_MS) {
            reporter.printProgress();
            lastProgressTime = currentTime;
        }
    }

    private void advanceToNextStep(long currentTime) {
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

    private void finishScenario() {
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
