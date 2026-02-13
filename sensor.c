#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int STATUS_OK = 0;
const int STATUS_FAILED = 1;
const int STATUS_STOPPED = 2;

const int MAX_NUMBER_OF_SAMPLES = 30;

int readSensor(int * value)
{
    static int v = 0;
    *value = v;
    v++;
    return (v > MAX_NUMBER_OF_SAMPLES) ? STATUS_STOPPED : STATUS_OK;
}

char** messages = 0;

static const int VALUE_LOW = 0;
static const int VALUE_HIGH = 1;
static const int ERROR = 2;

void initialize()
{
    if (messages == 0) {
        messages = (char**)malloc(sizeof(char*) * 3);
        messages[0] = (char*)malloc(128);
        messages[1] = (char*)malloc(128);
        messages[2] = (char*)malloc(128);
        strcpy(messages[VALUE_LOW], "Low");
        strcpy(messages[VALUE_HIGH], "High");
        strcpy(messages[ERROR], "Error occurred");
    }
}

void finalize()
{
    if (messages) {
        free(messages[0]);
        free(messages[1]);
        free(messages[2]);
    }
    free(messages);
}

void printMessage(int msgIndex, int value)
{
    const char* msg = messages[msgIndex];
    printf("Value: %d, State: %s\n", value, msg);
    fflush(stdout);
}

void reportSensorFailure()
{
    finalize();
    printMessage(ERROR, 0);
}

void handleSensorValue(int value)
{
    int index = -1;
    initialize();
    if (value >= 0 && value <= 10) {
        index = VALUE_LOW;
    } else if (value > 10 && value <= 20) {
        index = VALUE_HIGH;
    }
    printMessage(index, value);
}

void mainLoop()
{
    int sensorValue;
    int status = 1;
    while (1) {
        status = readSensor(&sensorValue);
        if (status == STATUS_STOPPED) {
            break;
        } else if (status == STATUS_FAILED) {
            reportSensorFailure();
            break;
        }
        handleSensorValue(sensorValue);
    }
    finalize();
}

int main()
{
    mainLoop();
    return 0;
}
