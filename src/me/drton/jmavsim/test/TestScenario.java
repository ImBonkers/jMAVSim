package me.drton.jmavsim.test;

import java.util.ArrayList;
import java.util.List;

/**
 * Data model for a test scenario.
 * Contains scenario metadata and list of test steps.
 */
public class TestScenario {
    private String name;
    private String description;
    private double globalTimeoutSeconds;
    private List<TestStep> steps;

    public TestScenario() {
        this.name = "unnamed";
        this.description = "";
        this.globalTimeoutSeconds = 300;  // 5 minutes default
        this.steps = new ArrayList<>();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public double getGlobalTimeoutSeconds() {
        return globalTimeoutSeconds;
    }

    public void setGlobalTimeoutSeconds(double globalTimeoutSeconds) {
        this.globalTimeoutSeconds = globalTimeoutSeconds;
    }

    public List<TestStep> getSteps() {
        return steps;
    }

    public void addStep(TestStep step) {
        steps.add(step);
    }

    public int getStepCount() {
        return steps.size();
    }

    @Override
    public String toString() {
        return "TestScenario[" + name + ", " + steps.size() + " steps]";
    }
}
