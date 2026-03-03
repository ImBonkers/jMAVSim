package me.drton.jmavsim.test;

import me.drton.jmavsim.test.steps.*;
import me.drton.jmavsim.test.steps.RebootStep;

import java.io.*;
import java.util.*;

/**
 * Parses JSON scenario files into TestScenario objects.
 * Uses manual JSON parsing to avoid external dependencies.
 */
public class ScenarioLoader {

    /**
     * Load a scenario from a JSON file
     */
    public static TestScenario load(String filePath) throws IOException {
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append("\n");
            }
        }
        return parse(content.toString());
    }

    /**
     * Parse JSON string into TestScenario
     */
    public static TestScenario parse(String json) throws IOException {
        TestScenario scenario = new TestScenario();

        try {
            Map<String, Object> root = parseObject(json.trim());

            if (root.containsKey("name")) {
                scenario.setName((String) root.get("name"));
            }
            if (root.containsKey("description")) {
                scenario.setDescription((String) root.get("description"));
            }
            if (root.containsKey("globalTimeoutSeconds")) {
                scenario.setGlobalTimeoutSeconds(toDouble(root.get("globalTimeoutSeconds")));
            }

            if (root.containsKey("steps")) {
                @SuppressWarnings("unchecked")
                List<Object> steps = (List<Object>) root.get("steps");
                for (Object stepObj : steps) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> stepMap = (Map<String, Object>) stepObj;
                    TestStep step = parseStep(stepMap);
                    if (step != null) {
                        scenario.addStep(step);
                    }
                }
            }
        } catch (Exception e) {
            throw new IOException("Failed to parse scenario JSON: " + e.getMessage(), e);
        }

        return scenario;
    }

    private static TestStep parseStep(Map<String, Object> stepMap) throws IOException {
        String type = (String) stepMap.get("type");
        if (type == null) {
            throw new IOException("Step missing 'type' field");
        }

        double timeout = stepMap.containsKey("timeoutSeconds") ?
                toDouble(stepMap.get("timeoutSeconds")) : 30.0;

        switch (type.toLowerCase()) {
            case "arm":
                return new ArmStep(timeout);

            case "disarm": {
                boolean force = stepMap.containsKey("force") &&
                        Boolean.TRUE.equals(stepMap.get("force"));
                return new DisarmStep(timeout, force);
            }

            case "takeoff": {
                double altitude = toDouble(stepMap.getOrDefault("altitude", 5.0));
                double tolerance = toDouble(stepMap.getOrDefault("tolerance", 0.5));
                return new TakeoffStep(altitude, tolerance, timeout);
            }

            case "hover": {
                double duration = toDouble(stepMap.getOrDefault("durationSeconds", 5.0));
                double maxDrift = toDouble(stepMap.getOrDefault("maxDrift", 0.5));
                return new HoverStep(duration, maxDrift, timeout);
            }

            case "goto": {
                double x = toDouble(stepMap.getOrDefault("x", 0.0));
                double y = toDouble(stepMap.getOrDefault("y", 0.0));
                double z = toDouble(stepMap.getOrDefault("z", 0.0));
                double tolerance = toDouble(stepMap.getOrDefault("tolerance", 1.0));
                return new GotoStep(x, y, z, tolerance, timeout);
            }

            case "land":
                return new LandStep(timeout);

            case "setwind": {
                double x = toDouble(stepMap.getOrDefault("x", 0.0));
                double y = toDouble(stepMap.getOrDefault("y", 0.0));
                double z = toDouble(stepMap.getOrDefault("z", 0.0));
                return new SetWindStep(x, y, z);
            }

            case "wait": {
                double duration = toDouble(stepMap.getOrDefault("durationSeconds", 1.0));
                return new WaitStep(duration);
            }

            case "reboot":
                return new RebootStep(timeout);

            default:
                throw new IOException("Unknown step type: " + type);
        }
    }

    private static double toDouble(Object value) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return Double.parseDouble(value.toString());
    }

    // Simple JSON parser implementation

    private static int pos;
    private static String jsonStr;

    private static Map<String, Object> parseObject(String json) {
        jsonStr = json;
        pos = 0;
        skipWhitespace();
        return readObject();
    }

    private static void skipWhitespace() {
        while (pos < jsonStr.length() && Character.isWhitespace(jsonStr.charAt(pos))) {
            pos++;
        }
    }

    private static char peek() {
        skipWhitespace();
        return pos < jsonStr.length() ? jsonStr.charAt(pos) : '\0';
    }

    private static char read() {
        skipWhitespace();
        return pos < jsonStr.length() ? jsonStr.charAt(pos++) : '\0';
    }

    private static Map<String, Object> readObject() {
        Map<String, Object> map = new LinkedHashMap<>();
        if (read() != '{') {
            throw new RuntimeException("Expected '{' at position " + pos);
        }

        skipWhitespace();
        if (peek() == '}') {
            pos++;
            return map;
        }

        while (true) {
            String key = readString();
            skipWhitespace();
            if (read() != ':') {
                throw new RuntimeException("Expected ':' at position " + pos);
            }
            Object value = readValue();
            map.put(key, value);

            skipWhitespace();
            char c = read();
            if (c == '}') break;
            if (c != ',') {
                throw new RuntimeException("Expected ',' or '}' at position " + pos);
            }
        }
        return map;
    }

    private static List<Object> readArray() {
        List<Object> list = new ArrayList<>();
        if (read() != '[') {
            throw new RuntimeException("Expected '[' at position " + pos);
        }

        skipWhitespace();
        if (peek() == ']') {
            pos++;
            return list;
        }

        while (true) {
            list.add(readValue());
            skipWhitespace();
            char c = read();
            if (c == ']') break;
            if (c != ',') {
                throw new RuntimeException("Expected ',' or ']' at position " + pos);
            }
        }
        return list;
    }

    private static String readString() {
        skipWhitespace();
        if (read() != '"') {
            throw new RuntimeException("Expected '\"' at position " + pos);
        }

        StringBuilder sb = new StringBuilder();
        while (pos < jsonStr.length()) {
            char c = jsonStr.charAt(pos++);
            if (c == '"') break;
            if (c == '\\' && pos < jsonStr.length()) {
                char next = jsonStr.charAt(pos++);
                switch (next) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    default: sb.append(next);
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static Object readValue() {
        skipWhitespace();
        char c = peek();

        if (c == '"') {
            return readString();
        } else if (c == '{') {
            return readObject();
        } else if (c == '[') {
            return readArray();
        } else if (c == 't' || c == 'f') {
            return readBoolean();
        } else if (c == 'n') {
            return readNull();
        } else if (c == '-' || Character.isDigit(c)) {
            return readNumber();
        } else {
            throw new RuntimeException("Unexpected character '" + c + "' at position " + pos);
        }
    }

    private static Boolean readBoolean() {
        if (jsonStr.substring(pos).startsWith("true")) {
            pos += 4;
            return true;
        } else if (jsonStr.substring(pos).startsWith("false")) {
            pos += 5;
            return false;
        }
        throw new RuntimeException("Expected boolean at position " + pos);
    }

    private static Object readNull() {
        if (jsonStr.substring(pos).startsWith("null")) {
            pos += 4;
            return null;
        }
        throw new RuntimeException("Expected null at position " + pos);
    }

    private static Number readNumber() {
        int start = pos;
        if (jsonStr.charAt(pos) == '-') pos++;

        while (pos < jsonStr.length() && Character.isDigit(jsonStr.charAt(pos))) {
            pos++;
        }

        boolean isFloat = false;
        if (pos < jsonStr.length() && jsonStr.charAt(pos) == '.') {
            isFloat = true;
            pos++;
            while (pos < jsonStr.length() && Character.isDigit(jsonStr.charAt(pos))) {
                pos++;
            }
        }

        if (pos < jsonStr.length() && (jsonStr.charAt(pos) == 'e' || jsonStr.charAt(pos) == 'E')) {
            isFloat = true;
            pos++;
            if (pos < jsonStr.length() && (jsonStr.charAt(pos) == '+' || jsonStr.charAt(pos) == '-')) {
                pos++;
            }
            while (pos < jsonStr.length() && Character.isDigit(jsonStr.charAt(pos))) {
                pos++;
            }
        }

        String numStr = jsonStr.substring(start, pos);
        if (isFloat) {
            return Double.parseDouble(numStr);
        } else {
            long val = Long.parseLong(numStr);
            if (val >= Integer.MIN_VALUE && val <= Integer.MAX_VALUE) {
                return (int) val;
            }
            return val;
        }
    }
}
