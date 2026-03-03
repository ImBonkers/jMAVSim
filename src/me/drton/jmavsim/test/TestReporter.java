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
    }

    /**
     * Print step start
     */
    public void printStepStart(TestStep step, int stepIndex) {
        // Format: [stepType] ...
        String name = step.getDisplayName();
        System.out.print(name + " ");
        progressDots = 0;
    }

    /**
     * Print progress dot (called periodically during step)
     */
    public void printProgress() {
        if (progressDots < MAX_PROGRESS_DOTS) {
            System.out.print(".");
            progressDots++;
        }
    }

    /**
     * Print step result
     */
    public void printStepResult(TestStep step, int stepIndex) {
        // Pad to align results
        int dots = MAX_PROGRESS_DOTS - progressDots;
        for (int i = 0; i < dots; i++) {
            System.out.print(".");
        }
        System.out.print(" ");

        double elapsed = step.getElapsedSeconds();

        if (step.isFailed()) {
            System.out.print("FAIL");
            if (step.getFailureReason() != null) {
                System.out.print(" (" + step.getFailureReason() + ")");
            }
            System.out.printf(" (%.1fs)%n", elapsed);
        } else {
            System.out.print("PASS");
            System.out.printf(" (%.1fs", elapsed);
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
