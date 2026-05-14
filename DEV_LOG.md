# Development Log - Homeostatic HRL Thesis

## Week 1-2: ROS2 & Gazebo Setup
**Date:** [Feb 24- March 9]

### What I Did
- Installed ROS2 Jazzy, Gazebo Harmonic
- Created homeostatic_bot package
- Built patrol_node (controls TurtleBot3)
- Created custom world with charging station
- Set up GitHub repository

### Decisions Made
- Used TwistStamped instead of Twist (Gazebo requirement)
- Created simple arena instead of complex world (controlled experiments)

### Problems & Solutions
| Problem | Solution |
|---------|----------|
| Robot not moving | Changed Twist to TwistStamped |
| Obstacles were black | Added diffuse color property |
| Multicast warnings | Set GZ_IP=127.0.0.1 |

### Resources Used
- TurtleBot3 Gazebo package
- ROS2 Jazzy documentation

--------------------------------

## Week 3 -4 : Energy Simulation
**Date:** [March 10]

### What I Did
- Created battery_node.py
- Simulates SOC drain when moving
- Simulates charging at charging station
- Updated battery_node for detection busing AR tag

### Decisions Made
- Used ROS2 battery node instead of LinearBatteryPlugin
- Reason: More control over degradation model, easier experimentation

### Problems & Solutions
| Problem | Solution |
|---------|----------|

### Next Steps
- Test battery node
- Add Huang et al. degradation model
