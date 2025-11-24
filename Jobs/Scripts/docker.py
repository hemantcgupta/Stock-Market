#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:21:15 2025

@author: hemantcgupta
"""

import docker
import docker.errors

def restart_docker_container(container_name):
    """
    Connects to the Docker daemon and restarts the specified container.
    """
    try:
        # 1. Initialize the Docker client (connects to the local Docker daemon)
        client = docker.from_env()
        print(f"Attempting to find container: {container_name}")
        # 2. Get the container object by name/ID
        container = client.containers.get(container_name)
        # 3. Restart the container
        print(f"Container '{container_name}' found. Restarting...")
        # You can specify a timeout in seconds (default is 10)
        container.restart(timeout=5) 
        print(f"✅ Container '{container_name}' successfully restarted.")
    except docker.errors.NotFound:
        print(f"❌ Error: Container '{container_name}' not found.")
    except docker.errors.APIError as e:
        print(f"❌ Error communicating with Docker API: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Replace this value with your actual container name or ID ---
CONTAINER_TO_RESTART = "sql2022"

restart_docker_container(CONTAINER_TO_RESTART)