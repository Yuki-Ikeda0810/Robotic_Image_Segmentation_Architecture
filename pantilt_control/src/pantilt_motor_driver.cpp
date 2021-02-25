#include "pantilt_motor_driver.hpp"

// Constructor
PanTiltMotorDriver::PanTiltMotorDriver():
baudrate_(BAUDRATE),
protocol_version_(PROTOCOL_VERSION){
}

// Destructor
PanTiltMotorDriver::~PanTiltMotorDriver(){
  closeDynamixel();
}

bool PanTiltMotorDriver::init(void){
  portHandler_   = dynamixel::PortHandler::getPortHandler(DEVICENAME);
  packetHandler_ = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  // Open port
  if (portHandler_->openPort()){
    printf("Succeeded to open the port!\n");
  }
  else{
    printf("Failed to open the port!\n");
    return false;
  }

  // Set port baudrate
  if (portHandler_->setBaudRate(baudrate_)){
    printf("Succeeded to change the baudrate!\n");
  }
  else{
    printf("Failed to change the baudrate!\n");
    return false;
  }

  // Enable Dynamixel Torque
  setTorque(PAN, true); setTorque(TILT, true);

  groupSyncWritePosition_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_X_GOAL_POSITION, LEN_X_GOAL_POSITION);
  groupSyncReadPosition_ = new dynamixel::GroupSyncRead(portHandler_, packetHandler_, ADDR_X_PRESENT_POSITION, LEN_X_PRESENT_POSITION);

  return true;
}

bool PanTiltMotorDriver::setTorque(uint8_t id, bool onoff){
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;

  dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_X_TORQUE_ENABLE, onoff, &dxl_error);
  if(dxl_comm_result != COMM_SUCCESS){
    packetHandler_->getTxRxResult(dxl_comm_result);
  }
  else if(dxl_error != 0){
    packetHandler_->getRxPacketError(dxl_error);
  }
  else{
    if(onoff == true) printf("Dynamixel ID:%03d has been successfully connected!\n", id);
    if(onoff == false) printf("Dynamixel ID:%03d has been successfully disconnected!\n", id);
  }
}

void PanTiltMotorDriver::closeDynamixel(void){
  // Disable Dynamixel Torque
  setTorque(PAN, false); setTorque(TILT, false);

  // Close port
  portHandler_->closePort();
}

bool PanTiltMotorDriver::controlPantilt(int32_t *value){
  bool dxl_addparam_result_;
  int8_t dxl_comm_result_;

  // Add parameter storage for Dynamixel goal position
  dxl_addparam_result_ = groupSyncWritePosition_->addParam(PAN, (uint8_t*)&value[0]);
  if (dxl_addparam_result_ != true)
    return false;

  dxl_addparam_result_ = groupSyncWritePosition_->addParam(TILT, (uint8_t*)&value[1]);
  if (dxl_addparam_result_ != true)
    return false;

  // Syncwrite goal position value
  dxl_comm_result_ = groupSyncWritePosition_->txPacket();
  if (dxl_comm_result_ != COMM_SUCCESS){
    packetHandler_->getTxRxResult(dxl_comm_result_);
    return false;
  }

  // Clear syncwrite parameter storage
  groupSyncWritePosition_->clearParam();
  return true;
}

bool PanTiltMotorDriver::addPresentParam(void){
  // Add parameter storage for Dynamixel present position
  bool dxl_addparam_result_ = false;

  dxl_addparam_result_ = groupSyncReadPosition_->addParam(PAN);
  if(dxl_addparam_result_ != true)
    return false;

  dxl_addparam_result_ = groupSyncReadPosition_->addParam(TILT);
  if(dxl_addparam_result_ != true)
    return false;

  return true;
}

int32_t PanTiltMotorDriver::feedbackPantilt(uint8_t id){
  int dxl_comm_result_ = COMM_TX_FAIL;
  uint8_t dxl_error_ = 0;
  bool dxl_getdata_result_ = false;
  int32_t steer_present_angle_;

  // Syncread present position
  dxl_comm_result_ = groupSyncReadPosition_->txRxPacket();
  if (dxl_comm_result_ != COMM_SUCCESS){
    printf("%s\n", packetHandler_->getTxRxResult(dxl_comm_result_));
  }
  else if (groupSyncReadPosition_->getError(id, &dxl_error_)){
    printf("[ID:%03d] %s\n", id, packetHandler_->getRxPacketError(dxl_error_));
  }

  // Check if groupsyncread data of Dynamixel is available
  dxl_getdata_result_ = groupSyncReadPosition_->isAvailable(id, ADDR_X_PRESENT_POSITION, LEN_X_PRESENT_POSITION);
  if (dxl_getdata_result_ != true){
    printf("[ID:%03d] groupSyncRead getdata failed\n", id);
    return 0;
  }

  // Get Dynamixel present position value
  steer_present_angle_ = groupSyncReadPosition_->getData(id, ADDR_X_PRESENT_POSITION, LEN_X_PRESENT_POSITION);

  return steer_present_angle_;
}

int32_t *PanTiltMotorDriver::setPantiltPosition(int32_t pan, int32_t tilt){
  // int32_t pantilt_position[2] = {0, };
  // int32_t pantilt_position[2];

  pantilt_position[0] = pan + 2048;
  pantilt_position[1] = tilt + 2048 - 300; // -150

  return pantilt_position;
}