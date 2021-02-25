#ifndef PANTILT_MOTOR_DRIVER_H_
#define PANTILT_MOTOR_DRIVER_H_

#include <ros/ros.h>
#include <stdio.h> // printf etc.
#include <dynamixel_sdk/dynamixel_sdk.h> // Uses Dynamixel SDK library

// Control table address (Dynamixel X-series)
#define ADDR_X_TORQUE_ENABLE            64
#define ADDR_X_GOAL_POSITION            116
#define ADDR_X_PRESENT_POSITION         132

// Data Byte Length
#define LEN_X_GOAL_POSITION             4
#define LEN_X_PRESENT_POSITION          4

// Dynamixel protocol version 2.0
#define PROTOCOL_VERSION                2.0

// Default setting
#define PAN                             16             // Dynamixel ID:14
#define TILT                            17             // Dynamixel ID:15

#define BAUDRATE                        3000000        // baud rate of Dynamixel
#define DEVICENAME                      "/dev/input/arm_pantilt"

#define TORQUE_ENABLE                   1              // Value for enabling the torque
#define TORQUE_DISABLE                  0              // Value for disabling the torque
#define DXL_MOVING_STATUS_THRESHOLD     10             // Dynamixel moving status threshold  // old param : 10

class PanTiltMotorDriver{
  public:
    PanTiltMotorDriver();
    ~PanTiltMotorDriver();
    bool init(void);
    void closeDynamixel(void);
    bool setTorque(uint8_t id, bool onoff);
    bool controlPantilt(int32_t *value);
    bool addPresentParam(void);
    int32_t feedbackPantilt(uint8_t);
    int32_t *setPantiltPosition(int32_t pan, int32_t tilt);

  private:
    uint32_t baudrate_;
    float  protocol_version_;
  
    int32_t pantilt_position[2] = {0, };

    // Initialize PortHandler instance
    // Set the port path
    // Get methods and members of PortHandlerLinux
    dynamixel::PortHandler *portHandler_;

    // Initialize PacketHandler instance
    // Set the protocol version
    // Get methods and members of Protocol2PacketHandler
    dynamixel::PacketHandler *packetHandler_;

    // Initialize GroupSyncWrite instance
    dynamixel::GroupSyncWrite *groupSyncWritePosition_;

    // Initialize GroupSyncWrite instance
    dynamixel::GroupSyncWrite *groupSyncWriteVelocity_;

    // Initialize GroupSyncRead instance
    dynamixel::GroupSyncRead *groupSyncReadPosition_;

    // Initialize GroupSyncRead instance
    dynamixel::GroupSyncRead *groupSyncReadVelocity_;
};

#endif // PANTILT_MOTOR_DRIVER_H_
