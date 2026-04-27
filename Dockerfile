FROM osrf/ros:humble-desktop-full

# Setup user for GUI to work without xhost
ARG UNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} -o ${UNAME}
RUN useradd -m -u ${UID} -g ${GID} -o -s /bin/bash ${UNAME}
RUN usermod -aG sudo ${UNAME}
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${UNAME}

# Disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

# Make default shell in Dockerfile bash instead of sh
SHELL ["/bin/bash", "-c"]

# Install Gazebo
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-nav2-simple-commander \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-turtlebot3-gazebo \
    ros-${ROS_DISTRO}-turtlebot3-navigation2 \
    ros-${ROS_DISTRO}-turtlebot3-description \
    ros-${ROS_DISTRO}-cv-bridge \
    python3-pip && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    python3-pil \
    && sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo python3 -m pip install --no-cache-dir langgraph langchain langchain-core langchain-openai langchain-anthropic langchain-mistralai langchain-ollama

# Source ROS workspace automatically when new terminal is opened
RUN echo ". /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "[ -f /workspace/install/setup.bash ] && . /workspace/install/setup.bash" >> ~/.bashrc && \
    echo "export GAZEBO_MODEL_PATH=/workspace/world_files/gazebo_models_worlds_collection/models/" >> ~/.bashrc

# Source ROS in the main terminal
COPY ros_entrypoint.sh /ros_entrypoint.sh

# Source ROS in the main terminal
ENTRYPOINT ["/ros_entrypoint.sh"]

# set local working directory
WORKDIR /workspace

CMD ["bash"]
