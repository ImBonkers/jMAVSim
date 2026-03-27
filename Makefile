# jMAVSim Build & Run
# Usage: make help

JAR        := out/production/jmavsim_run.jar
SERIAL     ?= /dev/ttyACM1
BAUD       ?= 921600
SCENARIO   ?= scenarios/simple_takeoff.json
JAVA_FLAGS := --add-opens java.desktop/sun.awt=ALL-UNNAMED \
              --add-opens java.desktop/sun.java2d=ALL-UNNAMED \
              --enable-native-access=ALL-UNNAMED

.PHONY: all build run test clean help

## Build fat jar with all dependencies
all: build

build:
	ant create_run_jar

## Run simulator with serial HIL (no scenario)
run: build
	java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD)

## Run test scenario (default: simple_takeoff)
##   make test SCENARIO=scenarios/basic_flight.json
test: build
	java $(JAVA_FLAGS) -jar $(JAR) -serial $(SERIAL) $(BAUD) \
		-test $(SCENARIO) -test-keep-running

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

## Clean build artifacts
clean:
	ant clean

help:
	@echo "jMAVSim Build & Run"
	@echo ""
	@echo "  make build          Build fat jar (jmavsim_run.jar)"
	@echo "  make run            Run simulator (serial HIL, no scenario)"
	@echo "  make test           Run test scenario (SCENARIO=path)"
	@echo "  make test-all       Run all scenarios sequentially"
	@echo "  make test-takeoff   Run simple_takeoff scenario"
	@echo "  make test-basic     Run basic_flight scenario"
	@echo "  make test-waypoint  Run waypoint_mission scenario"
	@echo "  make test-wind      Run wind_disturbance scenario"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  SERIAL=/dev/ttyACM1   Serial port (default: /dev/ttyACM1)"
	@echo "  BAUD=921600           Baud rate (default: 921600)"
	@echo "  SCENARIO=path.json    Scenario file for 'make test'"
