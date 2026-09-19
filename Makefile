# jMAVSim Build & Run
# Usage: make help

JAR        := out/production/jmavsim_run.jar
SERIAL     ?= /dev/ttyACM1
BAUD       ?= 921600
SCENARIO   ?= scenarios/simple_takeoff.json
OUTPUT     ?= logs
ROUTINE    ?= routines/quick_smoke.json
CSV        ?=
JAVA_FLAGS := --add-opens java.desktop/sun.awt=ALL-UNNAMED \
              --add-opens java.desktop/sun.java2d=ALL-UNNAMED \
              --enable-native-access=ALL-UNNAMED

.PHONY: all build run replay test test-all test-suite test-dry clean help

## Build fat jar with all dependencies
all: build

build:
	ant create_run_jar

## Run simulator with serial HIL (no scenario)
run: build
	java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD)

## Replay a CSV flight log in 3D viewer
##   make replay CSV=logs/.../flight.csv
replay: build
	@if [ -z "$(CSV)" ]; then echo "Usage: make replay CSV=path/to/flight.csv"; exit 1; fi
	java $(JAVA_FLAGS) -jar $(JAR) -replay $(CSV)

## Run test scenario interactively (keeps simulator open after test)
##   make test SCENARIO=scenarios/basic_flight.json
test: build
	java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD) \
		-test $(SCENARIO) -test-keep-running

## Run test scenario in batch mode (exits when done, returns exit code)
##   make test-batch SCENARIO=scenarios/basic_flight.json
test-batch: build
	java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD) \
		-no-gui -test $(SCENARIO) -test-output $(OUTPUT)

## Same as test-batch but with the 3D window visible, for demos/recording.
## Still exits when the scenario finishes so a campaign can continue.
test-batch-gui: build
	java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD) \
		-test $(SCENARIO) -test-output $(OUTPUT)

## Run all test scenarios sequentially
test-all: build
	@for s in scenarios/*.json; do \
		echo ""; \
		echo "========== Running: $$s =========="; \
		java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD) \
			-test $$s || echo "SCENARIO FAILED: $$s"; \
	done

## Shortcut targets for each scenario
test-takeoff: SCENARIO = scenarios/simple_takeoff.json
test-takeoff: test

test-basic: SCENARIO = scenarios/basic_flight.json
test-basic: test

test-waypoint: SCENARIO = scenarios/waypoint_mission.json
test-waypoint: test

test-wind: SCENARIO = scenarios/wind_disturbance.json
test-wind: test

## Run automated test suite from a routine file
##   make test-suite ROUTINE=routines/npu_board_regression.json
test-suite: build
	tools/test_runner/.venv/bin/python3 tools/test_runner/run_tests.py $(ROUTINE)

## Run automated test suite with the 3D window visible (demo/recording)
##   make test-suite-gui ROUTINE=routines/demo_showcase.json
test-suite-gui: build
	tools/test_runner/.venv/bin/python3 tools/test_runner/run_tests.py \
		--gui $(ROUTINE)

## Run full N6 regression (all 5 duty levels, n=10 each) — ~25 hours
test-n6-full: build
	tools/test_runner/.venv/bin/python3 tools/test_runner/run_tests.py \
		routines/npu_v2_0pct.json \
		routines/npu_v2_25pct.json \
		routines/npu_v2_50pct.json \
		routines/npu_v2_75pct.json \
		routines/npu_v2_100pct.json

## Run full Pixhawk baseline (n=10) — ~5 hours
test-pixhawk-full: build
	tools/test_runner/.venv/bin/python3 tools/test_runner/run_tests.py \
		routines/pixhawk6c_baseline.json

## Run everything: N6 full + Pixhawk baseline — ~30 hours
test-all-full: test-n6-full test-pixhawk-full

## Clean build artifacts
clean:
	ant clean

help:
	@echo "jMAVSim Build & Run"
	@echo ""
	@echo "  make build          Build fat jar (jmavsim_run.jar)"
	@echo "  make run            Run simulator (serial HIL, no scenario)"
	@echo "  make replay         Replay CSV flight log (CSV=path)"
	@echo "  make test           Run test scenario (SCENARIO=path)"
	@echo "  make test-all       Run all scenarios sequentially"
	@echo "  make test-suite     Run automated test suite (ROUTINE=path)"
	@echo "  make test-takeoff   Run simple_takeoff scenario"
	@echo "  make test-basic     Run basic_flight scenario"
	@echo "  make test-waypoint  Run waypoint_mission scenario"
	@echo "  make test-wind      Run wind_disturbance scenario"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  SERIAL=/dev/ttyACM1                   Serial port"
	@echo "  BAUD=921600                            Baud rate"
	@echo "  SCENARIO=path.json                     Scenario for 'make test'"
	@echo "  ROUTINE=routines/quick_smoke.json      Routine for 'make test-suite'"
	@echo "  CSV=path/to/flight.csv                 CSV file for 'make replay'"
