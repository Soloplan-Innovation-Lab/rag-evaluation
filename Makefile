.PHONY: start stop

PID_FILE := /workspace/tmp/fastapi_pids
LOG_DIR := /workspace/tmp

start:
	# Ensure the log directory exists
	mkdir -p $(LOG_DIR)
	# Create an empty PID file
	> $(PID_FILE)
	# Start the first uvicorn app and redirect output to log file
	cd /workspace/evaluation/api && setsid nohup uvicorn main:app --port 8801 > $(LOG_DIR)/eval_api.log 2>&1 & echo $$! >> $(PID_FILE) &
	# Start the second uvicorn app and redirect output to log file
	cd /workspace/rag-pipeline/api && setsid nohup uvicorn main:app --port 8802 > $(LOG_DIR)/rag_pipeline_api.log 2>&1 & echo $$! >> $(PID_FILE) &
	# Print a message
	echo "Uvicorn apps started. PIDs saved to $(PID_FILE)."

stop:
	# Stop the processes if the PID file exists
	if [ -f $(PID_FILE) ]; then \
		xargs -r kill < $(PID_FILE); \
		rm $(PID_FILE); \
	else \
		echo "No PID file found."; \
	fi
