#!/usr/bin/env bash
#
# Automated test runner for jMAVSim HIL scenarios.
# Reads a test routine JSON file and runs each scenario via 'make test-batch'.
#
# Usage:
#   ./tools/test_runner/run_tests.sh routines/quick_smoke.json
#   ./tools/test_runner/run_tests.sh routines/npu_board_regression.json -o logs/myrun
#   ./tools/test_runner/run_tests.sh routines/quick_smoke.json -v   # verbose (show sim output)
#
# Requires: jq (sudo apt install jq)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Parse arguments ---
ROUTINE=""
OUTPUT_OVERRIDE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            OUTPUT_OVERRIDE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <routine.json> [-o output_dir] [-v]"
            echo "  -o, --output  Custom output directory"
            echo "  -v, --verbose Show full simulator output"
            exit 0
            ;;
        *)
            ROUTINE="$1"
            shift
            ;;
    esac
done

if [ -z "$ROUTINE" ]; then
    echo "ERROR: No routine file specified"
    echo "Usage: $0 <routine.json> [-o output_dir] [-v]"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "ERROR: jq is required. Install with: sudo apt install jq"
    exit 1
fi

if [ ! -f "$ROUTINE" ]; then
    echo "ERROR: Routine file not found: $ROUTINE"
    exit 1
fi

# --- Helpers ---
format_duration() {
    local secs=$1
    if [ "$secs" -ge 3600 ]; then
        printf "%dh%02dm" $((secs/3600)) $(((secs%3600)/60))
    elif [ "$secs" -ge 60 ]; then
        printf "%dm%02ds" $((secs/60)) $((secs%60))
    else
        printf "%ds" "$secs"
    fi
}

format_eta() {
    local secs=$1
    date -d "+${secs} seconds" +%H:%M 2>/dev/null || date -v+${secs}S +%H:%M 2>/dev/null || echo "??:??"
}

progress_bar() {
    local current=$1
    local total=$2
    local width=30
    local filled=$((current * width / total))
    local empty=$((width - filled))
    local pct=$((current * 100 / total))
    printf "${CYAN}[${NC}"
    for ((b=0; b<filled; b++)); do printf "${GREEN}#${NC}"; done
    for ((b=0; b<empty; b++)); do printf "${DIM}-${NC}"; done
    printf "${CYAN}]${NC} %3d%%" "$pct"
}

# --- Read routine ---
BOARD=$(jq -r '.board' "$ROUTINE")
SERIAL=$(jq -r '.connection.port // "/dev/ttyACM0"' "$ROUTINE")
BAUD=$(jq -r '.connection.baudRate // 921600' "$ROUTINE")
PAUSE=$(jq -r '.pauseBetweenRunsSeconds // 5' "$ROUTINE")
DESCRIPTION=$(jq -r '.description // ""' "$ROUTINE")
VARS_JSON=$(jq -r '.variables // {}' "$ROUTINE")

# Output directory
if [ -n "$OUTPUT_OVERRIDE" ]; then
    OUTPUT_BASE="$OUTPUT_OVERRIDE"
else
    TS=$(date +%Y%m%d_%H%M%S)
    OUTPUT_BASE="logs/${BOARD}_${TS}"
fi

mkdir -p "$OUTPUT_BASE"
cp "$ROUTINE" "$OUTPUT_BASE/routine.json"

# --- Build run list and estimate times ---
NUM_SCENARIOS=$(jq '.scenarios | length' "$ROUTINE")
TOTAL_RUNS=0
TOTAL_ESTIMATED_SECS=0

# Arrays for scenario info
declare -a SCENARIO_FILES
declare -a SCENARIO_REPS
declare -a SCENARIO_TIMEOUTS
declare -a SCENARIO_NAMES

for ((i=0; i<NUM_SCENARIOS; i++)); do
    FILE=$(jq -r ".scenarios[$i].file" "$ROUTINE")
    REPS=$(jq -r ".scenarios[$i].repetitions // 1" "$ROUTINE")
    NAME=$(basename "$FILE" .json)

    # Read globalTimeoutSeconds from the scenario file for ETA
    if [ -f "$FILE" ]; then
        TIMEOUT=$(jq -r '.globalTimeoutSeconds // 300' "$FILE")
    else
        TIMEOUT=300
    fi

    SCENARIO_FILES+=("$FILE")
    SCENARIO_REPS+=("$REPS")
    SCENARIO_TIMEOUTS+=("$TIMEOUT")
    SCENARIO_NAMES+=("$NAME")

    TOTAL_RUNS=$((TOTAL_RUNS + REPS))
    TOTAL_ESTIMATED_SECS=$((TOTAL_ESTIMATED_SECS + TIMEOUT * REPS + PAUSE * REPS))
done

# --- Header ---
echo ""
echo -e "${BOLD}  jMAVSim Test Runner${NC}"
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"
echo -e "  Board:       ${BOLD}$BOARD${NC}"
echo -e "  Description: $DESCRIPTION"
echo -e "  Serial:      $SERIAL @ $BAUD"
echo -e "  Output:      $OUTPUT_BASE"
echo -e "  Scenarios:   $NUM_SCENARIOS scenarios, $TOTAL_RUNS total runs"
echo -e "  Max time:    $(format_duration $TOTAL_ESTIMATED_SECS)  (ETA $(format_eta $TOTAL_ESTIMATED_SECS))"
if [ "$VARS_JSON" != "{}" ]; then
    echo -e "  Variables:   $(echo "$VARS_JSON" | jq -r 'to_entries | map("\(.key)=\(.value)") | join(", ")')"
fi
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"
echo ""

# --- Counters ---
RUN_NUM=0
PASSED=0
FAILED=0
FAIL_LIST=""
ROUTINE_START=$(date +%s)

# Track actual durations for rolling ETA
declare -a RUN_DURATIONS

# --- Run function ---
run_scenario() {
    local SCENARIO="$1"
    local REP="$2"
    local SCENARIO_NAME
    SCENARIO_NAME=$(basename "$SCENARIO" .json)

    local START_EPOCH
    START_EPOCH=$(date +%s)
    local START_DATE
    START_DATE=$(date +%Y%m%d_%H%M%S)

    local RUN_DIR="$OUTPUT_BASE/.tmp_run"
    mkdir -p "$RUN_DIR"

    RUN_NUM=$((RUN_NUM + 1))
    local REP_FMT
    REP_FMT=$(printf "%02d" "$REP")

    # Calculate ETA based on remaining estimated time
    local ELAPSED=$((START_EPOCH - ROUTINE_START))
    local REMAINING_ESTIMATED=0
    if [ ${#RUN_DURATIONS[@]} -gt 0 ]; then
        # Use average actual duration for better ETA
        local SUM=0
        for d in "${RUN_DURATIONS[@]}"; do SUM=$((SUM + d)); done
        local AVG=$((SUM / ${#RUN_DURATIONS[@]}))
        REMAINING_ESTIMATED=$(( AVG * (TOTAL_RUNS - RUN_NUM + 1) ))
    else
        # No data yet, use timeout estimates for remaining
        local REMAINING_RUNS=$((TOTAL_RUNS - RUN_NUM + 1))
        REMAINING_ESTIMATED=$((TOTAL_ESTIMATED_SECS * REMAINING_RUNS / TOTAL_RUNS))
    fi

    # Status line
    echo -ne "\r\033[K"
    echo -e "  $(progress_bar $((RUN_NUM - 1)) $TOTAL_RUNS)  ${BOLD}$SCENARIO_NAME${NC} rep $REP  ${DIM}(${RUN_NUM}/${TOTAL_RUNS}, ETA $(format_eta $REMAINING_ESTIMATED))${NC}"

    # Apply variable substitution to scenario file
    local ACTUAL_SCENARIO="$SCENARIO"
    if [ "$VARS_JSON" != "{}" ]; then
        local TMP_SCENARIO="$OUTPUT_BASE/.tmp_scenario.json"
        cp "$SCENARIO" "$TMP_SCENARIO"
        for KEY in $(echo "$VARS_JSON" | jq -r 'keys[]'); do
            VALUE=$(echo "$VARS_JSON" | jq -r --arg k "$KEY" '.[$k]')
            sed -i "s/\"\\\$$KEY\"/$VALUE/g" "$TMP_SCENARIO"
        done
        ACTUAL_SCENARIO="$TMP_SCENARIO"
    fi

    local EXIT_CODE=0
    if [ "$VERBOSE" = true ]; then
        make test-batch SERIAL="$SERIAL" BAUD="$BAUD" SCENARIO="$ACTUAL_SCENARIO" OUTPUT="$RUN_DIR" || EXIT_CODE=$?
    else
        make test-batch SERIAL="$SERIAL" BAUD="$BAUD" SCENARIO="$ACTUAL_SCENARIO" OUTPUT="$RUN_DIR" > "$OUTPUT_BASE/.tmp_stdout.log" 2>&1 || EXIT_CODE=$?
    fi

    local END_EPOCH
    END_EPOCH=$(date +%s)
    local DURATION=$((END_EPOCH - START_EPOCH))
    RUN_DURATIONS+=("$DURATION")

    local END_DATE
    END_DATE=$(date +%Y%m%d_%H%M%S)

    local RESULT="PASS"
    local RESULT_COLOR="$GREEN"
    if [ "$EXIT_CODE" -eq 69 ]; then
        RESULT="ABORTED"
        RESULT_COLOR="$YELLOW"
        FAILED=$((FAILED + 1))
        FAIL_LIST="${FAIL_LIST}\n    ${YELLOW}x${NC} ${SCENARIO_NAME} rep ${REP} (aborted early)"
    elif [ "$EXIT_CODE" -ne 0 ]; then
        RESULT="FAIL"
        RESULT_COLOR="$RED"
        FAILED=$((FAILED + 1))
        FAIL_LIST="${FAIL_LIST}\n    ${RED}x${NC} ${SCENARIO_NAME} rep ${REP}"
    else
        PASSED=$((PASSED + 1))
    fi

    # Overwrite the progress line with result
    echo -ne "\r\033[K"
    echo -e "  $(progress_bar $RUN_NUM $TOTAL_RUNS)  ${BOLD}$SCENARIO_NAME${NC} rep $REP  ${RESULT_COLOR}${RESULT}${NC}  ${DIM}$(format_duration $DURATION)${NC}"

    # Move and rename log files
    local FINAL_DIR="$OUTPUT_BASE/$BOARD/$SCENARIO_NAME"
    mkdir -p "$FINAL_DIR"
    local PREFIX="${BOARD}_${SCENARIO_NAME}_rep${REP_FMT}_${START_DATE}_${END_DATE}_${RESULT}"

    for f in "$RUN_DIR"/*; do
        [ -f "$f" ] || continue
        local EXT="${f##*.}"
        mv "$f" "$FINAL_DIR/${PREFIX}.${EXT}"
    done

    # Save stdout log too if not verbose
    if [ "$VERBOSE" = false ] && [ -f "$OUTPUT_BASE/.tmp_stdout.log" ]; then
        mv "$OUTPUT_BASE/.tmp_stdout.log" "$FINAL_DIR/${PREFIX}.log"
    fi

    rm -rf "$RUN_DIR"
}

# --- Execute all runs ---
for ((i=0; i<NUM_SCENARIOS; i++)); do
    SCENARIO_FILE="${SCENARIO_FILES[$i]}"
    REPS="${SCENARIO_REPS[$i]}"
    NAME="${SCENARIO_NAMES[$i]}"

    for ((rep=1; rep<=REPS; rep++)); do
        run_scenario "$SCENARIO_FILE" "$rep"

        # Pause between runs (not after the last)
        if [ "$RUN_NUM" -lt "$TOTAL_RUNS" ] && [ "$PAUSE" -gt 0 ]; then
            echo -ne "  ${DIM}pausing ${PAUSE}s...${NC}"
            sleep "$PAUSE"
            echo -ne "\r\033[K"
        fi
    done
done

# --- Final Summary ---
ROUTINE_END=$(date +%s)
TOTAL_DURATION=$((ROUTINE_END - ROUTINE_START))

echo ""
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"
echo -e "  ${BOLD}COMPLETE${NC} — $BOARD"
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"
echo -e "  Total:    $((PASSED + FAILED)) runs in $(format_duration $TOTAL_DURATION)"
echo -e "  Passed:   ${GREEN}$PASSED${NC}"
if [ "$FAILED" -gt 0 ]; then
    echo -e "  Failed:   ${RED}$FAILED${NC}"
    echo -e ""
    echo -e "  Failed tests:"
    echo -e "$FAIL_LIST"
fi
echo ""
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}ALL TESTS PASSED${NC}"
else
    echo -e "  ${RED}${BOLD}SOME TESTS FAILED${NC}"
fi
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"
echo -e "  ${DIM}Output: $OUTPUT_BASE${NC}"
echo ""

# Write machine-readable summary
ALL_PASSED="true"
[ "$FAILED" -gt 0 ] && ALL_PASSED="false"

cat > "$OUTPUT_BASE/test_summary.json" << EOF
{
  "board": "$BOARD",
  "description": "$DESCRIPTION",
  "serial": "$SERIAL",
  "baudRate": $BAUD,
  "routine": "$(basename "$ROUTINE")",
  "timestamp": "$(date -Iseconds)",
  "totalRuns": $((PASSED + FAILED)),
  "passed": $PASSED,
  "failed": $FAILED,
  "allPassed": $ALL_PASSED,
  "durationSeconds": $TOTAL_DURATION,
  "outputDir": "$OUTPUT_BASE"
}
EOF

[ "$FAILED" -eq 0 ] && exit 0 || exit 1
