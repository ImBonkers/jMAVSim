package me.drton.jmavsim.test.steps;

import me.drton.jmavsim.test.CommandSender;
import me.drton.jmavsim.test.TestStep;
import me.drton.jmavsim.test.VehicleState;

/**
 * Step that reads a parameter, verifies it can be read, and confirms round-trip.
 * Sends PARAM_REQUEST_READ, waits for PARAM_VALUE response.
 */
public class VerifyParamRoundtripStep extends TestStep {
    private static final long COMMAND_RESEND_INTERVAL_MS = 1000;

    private String paramId;
    private long lastCommandTime;
    private CommandSender commandSenderRef;

    public VerifyParamRoundtripStep(String paramId, double timeoutSeconds) {
        super("verifyParamRoundtrip", timeoutSeconds);
        this.paramId = paramId;
        this.lastCommandTime = 0;
    }

    @Override
    public void start(CommandSender commandSender, long currentTime) {
        super.start(commandSender, currentTime);
        this.commandSenderRef = commandSender;
        commandSender.clearLastParam();
        commandSender.requestParam(paramId);
        lastCommandTime = currentTime;
    }

    @Override
    public void update(CommandSender commandSender, VehicleState state, long currentTime) {
        if (currentTime - lastCommandTime > COMMAND_RESEND_INTERVAL_MS) {
            commandSender.requestParam(paramId);
            lastCommandTime = currentTime;
        }
    }

    @Override
    public boolean checkComplete(VehicleState state) {
        if (commandSenderRef == null) return false;
        String lastId = commandSenderRef.getLastParamId();
        if (lastId != null && lastId.equals(paramId)) {
            float value = commandSenderRef.getLastParamValue();
            markCompleted(paramId + "=" + value);
            return true;
        }
        return false;
    }

    @Override
    public String getDisplayName() {
        return "[verifyParamRoundtrip " + paramId + "]";
    }
}
