## Start Docker
docker compose up -d --build

## Build Package
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
colcon build --packages-select nav2_llm_demo
source install/setup.bash

## Terminal 1: Start Gazebo + TurtleBot3
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

## Terminal 2: Start Nav2 + AMCL
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True use_rviz:=False


## Terminal 3: Start Groq Decision Node
docker compose exec autonomous_pathing_llm bash
bash ./run_llm_nav.sh

## Terminal 4: Send Mission
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub --once /navigation_request std_msgs/msg/String "{data: 'Reach the far side of the obstacle course'}"

## Optional: watch llm nav node status
docker compose exec autonomous_pathing_llm bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /navigation_status
