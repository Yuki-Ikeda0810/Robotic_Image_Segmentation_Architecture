#include "pantilt_motor_driver.hpp"
#include <pantilt_control/PanTilt.h> 

PanTiltMotorDriver pantilt_motor_driver;

// Pantilt control server
bool pantilt_server(pantilt_control::PanTilt::Request &req,
                     pantilt_control::PanTilt::Response &res){

  int32_t pan_present_position;
  int32_t tilt_present_position;

  // Write goal position value
  pantilt_motor_driver.controlPantilt(pantilt_motor_driver.setPantiltPosition(req.pan, req.tilt));

  // Continue until the pantilt position reaches the goal
  do{
    pan_present_position = pantilt_motor_driver.feedbackPantilt(PAN);
    tilt_present_position = pantilt_motor_driver.feedbackPantilt(TILT);

  }while((abs(req.pan + 2048 - pan_present_position) > DXL_MOVING_STATUS_THRESHOLD) &&
         (abs(req.tilt + 2048 - tilt_present_position) > DXL_MOVING_STATUS_THRESHOLD));

  res.success = true;
  return true;
}

// main
int main(int argc, char **argv){
  ros::init(argc, argv, "SSA_server");
  ros::NodeHandle nh;

  pantilt_motor_driver.init();
  pantilt_motor_driver.addPresentParam();
  pantilt_motor_driver.controlPantilt(pantilt_motor_driver.setPantiltPosition(0, 0));

  ros::ServiceServer service = nh.advertiseService("pantilt_control", pantilt_server);
  ROS_INFO("Ready to Pantilt control");
  ros::spin();

  pantilt_motor_driver.closeDynamixel();  

  return 0;
}