FROM osrf/ros:jazzy-desktop-full

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
RUN sudo apt-get update && sudo apt-get install ros-${ROS_DISTRO}-ros-gz

# Install dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    && sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Source ROS workspace automatically when new terminal is opened
RUN echo ". /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "[ -f /workspace/install/setup.bash ] && . /workspace/install/setup.bash" >> ~/.bashrc

# Source ROS in the main terminal
COPY ros_entrypoint.sh /ros_entrypoint.sh

# Source ROS in the main terminal
ENTRYPOINT ["/ros_entrypoint.sh"]

# set local working directory
WORKDIR /workspace

CMD ["bash"]
