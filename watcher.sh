#!/usr/bin/env bash

daemon=$1

# Catch SIGTERM and stop daemon correctly
_term() {
  $daemon stop
  exit 0
}

# Set trap for SIGTERM
trap _term SIGTERM

# Start daemon
$daemon start

# Wait for SIGTERM
sleep infinity

