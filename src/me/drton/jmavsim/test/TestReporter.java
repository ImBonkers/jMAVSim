package me.drton.jmavsim.test;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Handles test reporting: console output during run and JSON summary on completion.
 */
public class TestReporter {
    private TestScenario scenario;
    private String outputDir;
    private long scenarioStartTime;
    private String boardInfo;

    // Step results
    private List<StepResult> results;

    // Progress tracking
    private int progressDots;
    private static final int MAX_PROGRESS_DOTS = 20;

    // Live tree rendering
    private static final String C_RESET = "\u001b[0m";
    private static final String C_GREEN = "\u001b[32m";
    private static final String C_RED = "\u001b[31m";
    private static final String C_DIM = "\u001b[2m";
    private static final String C_CYAN = "\u001b[36m";

    private static final int PENDING = 0;
    private static final int ACTIVE = 1;
    private static final int PASSED = 2;
    private static final int FAILED = 3;

    private String[] treeNames;
    private int[] treeState;
    private String[] treeDetail;
    private boolean treeActive;
    private boolean treeDrawn;
    private boolean inRedraw;
    private PrintStream realOut;

    public TestReporter(TestScenario scenario, String outputDir) {
        this.scenario = scenario;
        this.outputDir = outputDir;
        this.scenarioStartTime = 0;
        this.boardInfo = "Unknown";
        this.results = new ArrayList<>();
        this.progressDots = 0;
    }

    public void setBoardInfo(String info) {
        this.boardInfo = info;
    }

    /**
     * Render the step tree.  Called once, then redrawn in place as steps
     * change state, so the whole scenario shape is visible from the start.
     */
    private void drawTree() {
        inRedraw = true;
        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < treeNames.length; i++) {
            boolean last = (i == treeNames.length - 1);
            String branch = last ? "\u2514\u2500 " : "\u251c\u2500 ";
            String mark;
            String colour;

            switch (treeState[i]) {
            case PASSED:
                mark = "\u2714";
                colour = C_GREEN;
                break;
            case FAILED:
                mark = "\u2718";
                colour = C_RED;
                break;
            case ACTIVE:
                mark = "\u25b6";
                colour = C_CYAN;
                break;
            default:
                mark = " ";
                colour = C_DIM;
                break;
            }

            sb.append(colour).append(branch).append(mark).append(' ')
              .append(treeNames[i]);

            if (treeDetail[i] != null) {
                sb.append("  ").append(treeDetail[i]);
            }

            sb.append(C_RESET).append("\u001b[K").append(System.lineSeparator());
        }

        realOut.print(sb);
        realOut.flush();
        treeDrawn = true;
        inRedraw = false;
    }

    /** Move the cursor back over the tree so the next draw overwrites it. */
    private void redrawTree() {
        if (!treeActive || !treeDrawn) {
            return;
        }

        realOut.print("\u001b[" + treeNames.length + "A");
        drawTree();
    }

    /**
     * Wrap stdout so output from other components (command sender, serial
     * port, ...) appears above the tree instead of corrupting it.
     */
    private void interceptStdout() {
        realOut = System.out;

        OutputStream sink = new OutputStream() {
            private StringBuilder line = new StringBuilder();

            @Override
            public void write(int b) {
                if (inRedraw) {
                    realOut.write(b);
                    return;
                }

                if (b == '\n') {
                    String msg = line.toString();
                    line.setLength(0);

                    if (msg.trim().isEmpty()) {
                        return;
                    }

                    if (treeDrawn) {
                        realOut.print("\u001b[" + treeNames.length + "A");
                    }

                    realOut.print(msg + "\u001b[K" + System.lineSeparator());
                    drawTree();
                } else if (b != '\r') {
                    line.append((char) b);
                }
            }
        };

        System.setOut(new PrintStream(sink, true));
    }

    /** Restore the original stdout. */
    private void releaseStdout() {
        if (realOut != null) {
            System.setOut(realOut);
            treeActive = false;
        }
    }

    /**
     * Print scenario header
     */
    public void printHeader() {
        scenarioStartTime = System.currentTimeMillis();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String timestamp = sdf.format(new Date(scenarioStartTime));

        System.out.println();
        System.out.println("=== SCENARIO: " + scenario.getName() + " ===");
        System.out.println("Board: " + boardInfo);
        System.out.println("Date: " + timestamp);
        System.out.println();

        List<TestStep> steps = scenario.getSteps();
        treeNames = new String[steps.size()];
        treeState = new int[steps.size()];
        treeDetail = new String[steps.size()];

        for (int i = 0; i < steps.size(); i++) {
            treeNames[i] = steps.get(i).getDisplayName();
            treeState[i] = PENDING;
        }

        // Only render the live tree on an interactive terminal.  When output
        // is redirected (batch runs, the campaign runner) the cursor-up
        // redraws would pile up in the log file, so fall back to plain
        // line-by-line output instead.
        treeActive = (System.console() != null);

        if (treeActive) {
            interceptStdout();
            drawTree();
        }
    }

    /**
     * Print step start
     */
    public void printStepStart(TestStep step, int stepIndex) {
        progressDots = 0;

        if (treeActive && stepIndex < treeState.length) {
            treeState[stepIndex] = ACTIVE;
            treeNames[stepIndex] = step.getDisplayName();
            redrawTree();
            return;
        }

        System.out.print(step.getDisplayName() + " ");
    }

    /** Index of the step currently marked active, or -1. */
    private int activeIndex() {
        for (int i = 0; i < treeState.length; i++) {
            if (treeState[i] == ACTIVE) {
                return i;
            }
        }

        return -1;
    }

    /**
     * Print progress dot (called periodically during step)
     */
    public void printProgress() {
        if (treeActive) {
            return;
        }

        if (progressDots < MAX_PROGRESS_DOTS) {
            System.out.print(".");
            progressDots++;
        }
    }

    /**
     * Print progress string with telemetry (replaces dot)
     */
    public void printProgressString(String progress) {
        if (treeActive) {
            int idx = activeIndex();

            if (idx >= 0) {
                treeDetail[idx] = C_DIM + progress + C_RESET;
                redrawTree();
            }

            return;
        }

        System.out.print("\r  " + progress);
        // Don't increment progressDots — these overwrite in place
    }

    /**
     * Print step result
     */
    public void printStepResult(TestStep step, int stepIndex) {
        String name = step.getDisplayName();
        double elapsed = step.getElapsedSeconds();

        if (treeActive && stepIndex < treeState.length) {
            treeState[stepIndex] = step.isFailed() ? FAILED : PASSED;
            treeNames[stepIndex] = name;

            StringBuilder d = new StringBuilder();
            d.append(String.format("%.1fs", elapsed));

            String extra = step.isFailed() ? step.getFailureReason()
                                           : step.getResultDetails();

            if (extra != null) {
                d.append(", ").append(extra);
            }

            treeDetail[stepIndex] = C_DIM + d + C_RESET;
            redrawTree();

            StepResult tr = new StepResult();
            tr.stepIndex = stepIndex;
            tr.stepType = step.getType();
            tr.displayName = name;
            tr.passed = !step.isFailed();
            tr.elapsedSeconds = elapsed;
            tr.details = step.isFailed() ? step.getFailureReason()
                                         : step.getResultDetails();
            results.add(tr);
            return;
        }

        // Clear any carriage-return progress line, then print result on new line
        System.out.print("\r                                                                      \r");

        if (step.isFailed()) {
            System.out.printf("%s FAIL", name);
            if (step.getFailureReason() != null) {
                System.out.print(" (" + step.getFailureReason() + ")");
            }
            System.out.printf(" (%.1fs)%n", elapsed);
        } else {
            System.out.printf("%s PASS (%.1fs", name, elapsed);
            if (step.getResultDetails() != null) {
                System.out.print(", " + step.getResultDetails());
            }
            System.out.println(")");
        }

        // Record result
        StepResult result = new StepResult();
        result.stepIndex = stepIndex;
        result.stepType = step.getType();
        result.displayName = step.getDisplayName();
        result.passed = !step.isFailed();
        result.elapsedSeconds = elapsed;
        result.details = step.isFailed() ? step.getFailureReason() : step.getResultDetails();
        results.add(result);
    }

    /**
     * Print for instantaneous steps (like setWind)
     */
    public void printInstantStep(TestStep step, int stepIndex) {
        String name = step.getDisplayName();
        System.out.println(name + " OK");

        StepResult result = new StepResult();
        result.stepIndex = stepIndex;
        result.stepType = step.getType();
        result.displayName = step.getDisplayName();
        result.passed = true;
        result.elapsedSeconds = 0;
        result.details = null;
        results.add(result);
    }

    /**
     * Print summary and write JSON report
     */
    public void printSummary() {
        releaseStdout();

        long endTime = System.currentTimeMillis();
        double totalTime = (endTime - scenarioStartTime) / 1000.0;

        int passed = 0;
        int total = results.size();
        for (StepResult r : results) {
            if (r.passed) passed++;
        }

        System.out.println();
        if (passed == total) {
            System.out.println("RESULT: " + passed + "/" + total + " PASSED");
        } else {
            System.out.println("RESULT: " + passed + "/" + total + " PASSED, " + (total - passed) + " FAILED");
        }
        System.out.printf("Total time: %.1fs%n", totalTime);
        System.out.println();

        // Write JSON report if output directory specified
        if (outputDir != null && !outputDir.isEmpty()) {
            writeJsonReport(totalTime, passed, total);
        }
    }

    private void writeJsonReport(double totalTime, int passed, int total) {
        File dir = new File(outputDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        SimpleDateFormat fileSdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String fileName = scenario.getName() + "_" + fileSdf.format(new Date(scenarioStartTime)) + ".json";
        File reportFile = new File(dir, fileName);

        try (PrintWriter writer = new PrintWriter(new FileWriter(reportFile))) {
            writer.println("{");
            writer.println("  \"scenario\": \"" + escapeJson(scenario.getName()) + "\",");
            writer.println("  \"description\": \"" + escapeJson(scenario.getDescription()) + "\",");
            writer.println("  \"board\": \"" + escapeJson(boardInfo) + "\",");
            writer.println("  \"timestamp\": \"" + new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss").format(new Date(scenarioStartTime)) + "\",");
            writer.println("  \"totalTimeSeconds\": " + String.format("%.2f", totalTime) + ",");
            writer.println("  \"passed\": " + passed + ",");
            writer.println("  \"failed\": " + (total - passed) + ",");
            writer.println("  \"total\": " + total + ",");
            writer.println("  \"success\": " + (passed == total) + ",");
            writer.println("  \"steps\": [");

            for (int i = 0; i < results.size(); i++) {
                StepResult r = results.get(i);
                writer.print("    {");
                writer.print("\"index\": " + r.stepIndex + ", ");
                writer.print("\"type\": \"" + escapeJson(r.stepType) + "\", ");
                writer.print("\"name\": \"" + escapeJson(r.displayName) + "\", ");
                writer.print("\"passed\": " + r.passed + ", ");
                writer.print("\"elapsedSeconds\": " + String.format("%.2f", r.elapsedSeconds));
                if (r.details != null) {
                    writer.print(", \"details\": \"" + escapeJson(r.details) + "\"");
                }
                writer.print("}");
                if (i < results.size() - 1) {
                    writer.println(",");
                } else {
                    writer.println();
                }
            }

            writer.println("  ]");
            writer.println("}");

            System.out.println("Report written to: " + reportFile.getAbsolutePath());

        } catch (IOException e) {
            System.err.println("Failed to write report: " + e.getMessage());
        }
    }

    private String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /**
     * Get exit code based on results (0 = all passed, 1 = some failed)
     */
    public int getExitCode() {
        for (StepResult r : results) {
            if (!r.passed) return 1;
        }
        return 0;
    }

    // Internal class for tracking step results
    private static class StepResult {
        int stepIndex;
        String stepType;
        String displayName;
        boolean passed;
        double elapsedSeconds;
        String details;
    }
}
