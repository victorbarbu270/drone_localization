# IMU Motion Data Collection

This project collects labeled IMU data using an Arduino Nano 33 BLE Sense REV2 for training a TinyML localization model. The data is collected by moving the board by hand to simulate different motion states.

## Hardware Requirements
- Arduino Nano 33 BLE SENSE REV2
- SD Card Module
- USB cable for power
- (Optional) Small power bank for untethered operation

## Wiring
Connect the SD card module to the Arduino:
- CS -> D10
- SCK -> D13
- MOSI -> D11
- MISO -> D12
- VCC -> 3.3V
- GND -> GND

## Data Collection Protocol

### Motion States to Simulate
```
0 - STATIONARY_FLAT (board lying flat on table)
1 - STATIONARY_VERTICAL (board held vertically)
2 - MOVING_UP (smooth upward motion)
3 - MOVING_DOWN (smooth downward motion)
4 - HOVERING (trying to hold board as still as possible in air)
5 - MOVING_FORWARD (smooth forward motion)
6 - MOVING_BACKWARD (smooth backward motion)
7 - MOVING_LEFT (smooth left motion)
8 - MOVING_RIGHT (smooth right motion)
9 - ROTATING_CW (rotating clockwise while flat)
10 - ROTATING_CCW (rotating counter-clockwise while flat)
```

### Data Format
The data is saved in CSV format with the following columns:
```
timestamp,accel_x,accel_y,accel_z,state
```

### Collection Procedure
1. Mount the Arduino securely on a small rigid board or case
2. Connect power (USB or battery)
3. Wait for the "Card initialized" message on Serial monitor
4. For each motion state:
   - Hold the board in starting position
   - Perform the motion smoothly for 3-5 seconds
   - Try to maintain consistent speed
   - Repeat 3-5 times for variety

### Tips for Good Data Collection
1. Stationary States:
   - Use a flat surface for STATIONARY_FLAT
   - Use a wall or vertical surface for reference in STATIONARY_VERTICAL
   - Try to minimize hand shake in HOVERING

2. Linear Movements:
   - Move in straight lines
   - Keep consistent speed
   - Use markers or reference points for distance
   - Try to maintain the same orientation during movement

3. Rotations:
   - Mark start/end points for consistent rotation angles
   - Try to rotate in place without translation
   - Keep the rotation speed steady

### Manual State Changes
You can modify the code to:
1. Use a button to trigger state changes
2. Use serial commands to change states
3. Set fixed time intervals for each state

## Data Processing
1. After collecting data, remove the SD card
2. Each session will be in a separate numbered CSV file
3. Use these files for training your TinyML model

## Next Steps
- Analyze the data to verify clean state transitions
- Look for characteristic patterns in each motion state
- Consider collecting data at different speeds
- Try combining multiple motions
- Test in different orientations

## Tips for Later Drone Application
The handheld data can help develop initial algorithms, but keep in mind:
- Drone movements will be more precise
- Vibrations will be different
- Additional factors like wind will come into play
- You'll need to account for motor interference 