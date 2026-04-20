# TODO

- Add runtime obstacle updates so lidar or vision can mark graph edges `blocked` after the initial occupancy-map graph is built.
- Localize the robot to the nearest graph edge as well as the nearest node so replanning can resume from mid-corridor positions more accurately.
- Add a Nav2-backed node executor as an alternative to direct `cmd_vel` traversal once the graph-path architecture is stable.
- Visualize the deterministic topology graph in RViz or on the debug map image to inspect node placement and blocked-edge updates.
- Compare LLM route selection against deterministic shortest-path routing on the same extracted graph.
