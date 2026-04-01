## Working Startup

Run `run_llm_nav.sh` only inside the Docker container.
Do not run it from your Mac host shell.

## Terminal 1: Start Docker

From the repo root on your Mac:

```bash
docker compose up -d --build
```

## Terminal 2: Build The ROS Package

Open a new terminal on your Mac, then enter the container:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select nav2_llm_demo
source install/setup.bash
```

If the build succeeds, stay in this terminal if you want, or exit it.

## Terminal 3: Run Everything For Step 1-4

Open a new terminal on your Mac, then enter the container:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container:

```bash
bash ./run_llm_nav.sh
```

Or with a custom mission:

```bash
bash ./run_llm_nav.sh "Reach the far side of the obstacle course"
```

That script now does all of this:
- starts Gazebo + TurtleBot3
- starts Nav2 + AMCL
- starts `llm_nav_node`
- starts a background `ros2 topic echo /navigation_status`
- publishes one mission request to `/navigation_request`

Leave this terminal open while the system is running.

## Terminal 4: Optional Manual Mission Publish

Only open this if you want to send another mission while Terminal 3 is still running.

On your Mac:

```bash
docker compose exec autonomous_pathing_llm bash
```

Inside the container:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub --once /navigation_request std_msgs/msg/String "{data: 'Reach the loading zone'}"
```

## If It Fails Immediately

Check these first:
- you are inside the Docker container, not your Mac host shell
- `.env` exists at `/workspace/.env`
- `.env` contains `GROQ_API_KEY=...`
- `colcon build --packages-select nav2_llm_demo` finished successfully
- you left Terminal 3 open after starting `run_llm_nav.sh`

## Step 2: Start LangGraph controller
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch nav2_llm_tools step2_tools.launch.py

## Step 2: Start MCP server for an external LLM client
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run nav2_llm_tools mcp_server
