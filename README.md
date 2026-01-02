# Capstone-project

This project tackles the security challenge of unauthorized drone incursions into restricted airspace by developing a system that detects and tracks intruders using multi-sensor data. By employing GPS spoofing, the system subtly diverts drones away from no-fly zones and back onto their intended paths. A simulation-based approach ensures a non-disruptive, scalable solution for safeguarding sensitive areas.

Simulation setup-
1) Install Ubuntu 20.04.5 LTS
2) For installing ardupilot, Gazebo 11 and ROS Noetic, follow - https://github.com/SkyRats/sky_sim
3) Install MAVROS
4) Create virtual environments for the YOLOv8 deployment and bytetracker algorithm
5) Move the world file to the worlds folder in sky_ws directory and make a launch file for the same

Steps to execute - 

Terminal 1-

1) sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map
2) mode guided
3) arm throttle
4) takeoff 10
5) module load message
6) message SET_POSITION_TARGET_LOCAL_NED 0 0 0 1 3576 100 0 -10 0 0 0 0 0 0 0 0

Terminal 2-

1) roslaunch sky_sim my_custom_world.launch use_sim_time:=true

Terminal 3- 

1) roslaunch mavros px4.launch fcu_url:=udp://:14550@127.0.0.1:14550

Terminal 4-

1) source ~/yolo_env_py311/bin/activate
2) rosrun ml_detector run_yolo_node.sh _model_path:=/home/vtsv/sky_ws/src/ml_detector/models/latestyolo.pt _camera_topic:=/cam_south/cam_south/image_raw

Terminal 5-

1) conda activate drone_tracker
2) cd ~/sky_ws
3) source devel/setup.bash
4) rosrun ml_detector bytetracker_node.py

Terminal 6-

1) rosrun ml_detector tangent.py

Terminal 7-

1) rostopic echo mavros/local_position/odom
