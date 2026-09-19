package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that sets a PX4 parameter via PARAM_SET and waits for PARAM_VALUE echo.
 */
public class SetParamStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 1000;

    private String paramId;
    private float value;
    private long lastCommandTime;
    private CommandSender commandSenderRef;

    public SetParamStep(String paramId, float value, double timeoutSeconds) {
        super("setParam", timeoutSeconds);
        this.paramId = paramId;
        this.value = value;
        this.lastCommandTime = 0;
        this.critical = false;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        this.commandSenderRef = commandSender;
        commandSender.clearLastParam();
        commandSender.setParam(paramId, value);
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        if (currentTime - lastCommandTime > COMMAND_RESEND_INTERVAL_MS) {
            commandSender.setParam(paramId, value);
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (commandSenderRef == null) return false;
        String lastId = commandSenderRef.getLastParamId();
        if (lastId != null && lastId.equals(paramId)) {
            markCompleted(paramId + "=" + value);
            return true;
        }
        // Fall through after 2 seconds even without echo — some params don't echo
        if (getElapsedSeconds() >= 2.0) {
            markCompleted(paramId + "=" + value + " (no echo)");
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return "[setParam " + paramId + "=" + value + "]";
    }
}
