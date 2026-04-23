/*
 * NYU-FAMS-HRI-CV — Teensy basic chassis via CAN (FlexCAN_T4).
 *
 * Library: https://github.com/tonton81/FlexCAN_T4
 * Baud: 1 Mbps. CAN id 0x200, 8 bytes: int16 motor A, int16 motor B (big-endian high byte first).
 *
 * Serial (115200): lines FORWARD, BACK, LEFT, RIGHT, STOP (case-sensitive, newline-terminated).
 * Motor signs below match your drivetrain wiring; tune SPEED / delay as needed.
 * Python still sends the same tokens (see ``src/embodied_policy/emotion_to_position.py``).
 */

#include <FlexCAN_T4.h>

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;

int16_t m1 = 0;
int16_t m2 = 0;

#define SPEED 2500

void sendMotorCurrents(int16_t a, int16_t b) {
    CAN_message_t msg;
    msg.id = 0x200;
    msg.flags.extended = 0;
    msg.len = 8;

    msg.buf[0] = a >> 8;
    msg.buf[1] = a;
    msg.buf[2] = b >> 8;
    msg.buf[3] = b;
    msg.buf[4] = 0;
    msg.buf[5] = 0;
    msg.buf[6] = 0;
    msg.buf[7] = 0;

    can1.write(msg);
}

void handleCommand(String cmd) {
    cmd.trim();

    if (cmd == "FORWARD") {
        m1 = SPEED;
        m2 = -SPEED;
        sendMotorCurrents(m1, m2);
        delay(2000);
        m1 = 0;
        m2 = 0;
    } else if (cmd == "BACK") {
        m1 = -SPEED;
        m2 = SPEED;
        sendMotorCurrents(m1, m2);
        delay(2000);
        m1 = 0;
        m2 = 0;
    } else if (cmd == "LEFT") {
        m1 = -SPEED;
        m2 = -SPEED;
        sendMotorCurrents(m1, m2);
        delay(2000);
        m1 = 0;
        m2 = 0;
    } else if (cmd == "RIGHT") {
        m1 = SPEED;
        m2 = SPEED;
        sendMotorCurrents(m1, m2);
        delay(2000);
        m1 = 0;
        m2 = 0;
    } else if (cmd == "STOP") {
        m1 = 0;
        m2 = 0;
    }
}

void setup() {
    Serial.begin(115200);
    can1.begin();
    can1.setBaudRate(1000000);
    Serial.println("CAN Ready");
}

void loop() {
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        Serial.println("Got: " + cmd);
        handleCommand(cmd);
    }
    sendMotorCurrents(m1, m2);
    delay(10);
}
