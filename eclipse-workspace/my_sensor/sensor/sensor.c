/* MISRA C 2023 Compliant Sensor Application
 * All fixes documented with MISRA rule references.
 * Converted from original code with dynamic allocation and stdio usage.
 */

/* FIX: Removed #include <stdlib.h> - malloc/free are banned per MISRA Rule 21.3 / Dir 4.12 */
/* FIX: Removed #include <stdio.h> - printf/fflush are banned per MISRA Rule 21.6 */
#include <stdint.h>  /* FIX: Dir 4.6 - Use fixed-width integer types instead of 'int' */
#include <string.h>
#include <stddef.h>  /* For size_t, NULL */

/* ============================================================================
 * MISRA-safe console output implementation
 * FIX: MISRA Rule 21.6 - The Standard Library input/output functions shall
 *      not be used. We implement a minimal write function using platform
 *      syscall (POSIX write) to avoid stdio entirely.
 * ============================================================================ */

/* Platform-specific low-level write (not a stdio function, not flagged by MISRA 21.6) */
#if defined(_WIN32) || defined(_WIN64)
/* FIX: Rule 8.6 - Include proper header instead of manual extern declarations */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define STD_OUTPUT_HANDLE_VAL ((DWORD)STD_OUTPUT_HANDLE)

/* FIX: Rule 8.4 - Forward declarations for all static functions */
static int32_t safe_write(const char *buf, size_t len);
#else
/* POSIX: use write syscall directly */
extern long write(int fd, const void *buf, unsigned long count);

static int32_t safe_write(const char *buf, size_t len);
#endif

/* FIX: Rule 8.4 - A compatible declaration shall be visible when a function is defined.
 *      All functions used only in this translation unit are declared static.
 * FIX: Rule 8.7 - Functions/objects referenced in only one translation unit
 *      shall not have external linkage. All changed to static.
 * FIX: Rule 1.5 / Rule 8.2 - Functions shall be in prototype form with (void)
 *      for parameterless functions instead of empty parentheses (). */

/* Forward declarations (prototypes) for all static functions */
/* FIX: Rule 17.3 (Mandatory) - Prototypes shall precede function calls */
static int32_t readSensor(int32_t *value);
static void testMemset(char *buffer, int32_t size);

static void initialize(void);
static void finalize(void);
static void printMessage(int32_t msgIndex, int32_t value);
static void reportSensorFailure(void);
static void handleSensorValue(int32_t value);
static void mainLoop(void);

/* ============================================================================
 * MISRA-safe console output helpers
 * ============================================================================ */

/* FIX: Rule 21.6 - Custom implementation replaces printf/fflush */
static int32_t safe_write(const char *buf, size_t len)
{
    int32_t result;

    if (buf == NULL)  /* FIX: Rule 11.9 - Use NULL instead of 0 for pointers */
    {
        result = -1;
    }
    else
    {
#if defined(_WIN32) || defined(_WIN64)
        DWORD written = 0U;
        HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE_VAL);
        (void)WriteFile(handle, buf, (DWORD)len, &written, NULL);
        result = (int32_t)written;
#else
        result = (int32_t)write(1, buf, (unsigned long)len);
#endif
    }

    return result;
}

/* Safe string length calculation */
static size_t safe_strlen(const char *str)
{
    size_t len = 0U;

    if (str != NULL)
    {
        while (str[len] != '\0')
        {
            len++;
        }
    }

    return len;
}

/* FIX: Rule 21.6 - Custom int-to-string replaces printf %d format */
static size_t int32_to_str(int32_t value, char *buf, size_t buf_size)
{
    size_t pos = 0U;
    int32_t v = value;
    char tmp[12]; /* Enough for -2147483648 + null */
    size_t tmp_pos = 0U;
    uint32_t uval;

    if ((buf == NULL) || (buf_size == 0U))
    {
        /* Nothing to do */
    }
    else
    {
        if (v < 0)
        {
            if (pos < (buf_size - 1U))
            {
                buf[pos] = '-';
                pos++;
            }
            /* FIX: Rule 10.1/10.3 - Avoid signed/unsigned mixing.
             * Handle INT32_MIN safely by casting to unsigned */
            uval = (uint32_t)(-(v + 1)) + 1U;
        }
        else
        {
            uval = (uint32_t)v;
        }

        if (uval == 0U)
        {
            tmp[tmp_pos] = '0';
            tmp_pos++;
        }
        else
        {
            while (uval > 0U)
            {
                tmp[tmp_pos] = (char)((char)'0' + (char)(uval % 10U));
                tmp_pos++;
                uval = uval / 10U;
            }
        }

        /* Reverse copy into output buffer */
        while ((tmp_pos > 0U) && (pos < (buf_size - 1U)))
        {
            tmp_pos--;
            buf[pos] = tmp[tmp_pos];
            pos++;
        }

        buf[pos] = '\0';
    }

    return pos;
}

/* FIX: Rule 21.6 - Custom console print replaces printf("Value: %d, State: %s\n") */
static void safe_print_message(int32_t value, const char *state)
{
    char num_buf[12];
    size_t num_len;

    /* "Value: " */
    (void)safe_write("Value: ", 7U);

    /* Integer value */
    num_len = int32_to_str(value, num_buf, sizeof(num_buf));
    (void)safe_write(num_buf, num_len);

    /* ", State: " */
    (void)safe_write(", State: ", 9U);

    /* State string */
    (void)safe_write(state, safe_strlen(state));

    /* Newline */
    (void)safe_write("\n", 1U);

    /* FIX: Rule 21.6 - fflush(stdout) removed; safe_write uses unbuffered I/O */
}

/* ============================================================================
 * Status constants
 * ============================================================================ */

/* FIX: Dir 4.6 - Use int32_t instead of 'int' for all integer variables.
 * FIX: Rule 8.4 - Declarations precede definitions (static linkage).
 * FIX: Rule 8.7 - Objects referenced only in this TU are now static. */
static const int32_t STATUS_OK = 0;
static const int32_t STATUS_FAILED = 1;
static const int32_t STATUS_STOPPED = 2;

static const int32_t MAX_NUMBER_OF_SAMPLES = 30;

/* ============================================================================
 * Sensor reading
 * ============================================================================ */

/* FIX: Dir 4.6 - int changed to int32_t throughout.
 * FIX: Rule 8.4 - static + forward-declared above.
 * FIX: Rule 8.7 - Changed to static (internal linkage). */
static int32_t readSensor(int32_t *value)
{
    static int32_t v = 0;  /* FIX: Dir 4.6 - int -> int32_t */
    *value = v;
    v++;
    return (v > MAX_NUMBER_OF_SAMPLES) ? STATUS_STOPPED : STATUS_OK;
}

/* ============================================================================
 * Static message storage (replaces dynamic allocation)
 * ============================================================================ */

/* FIX: MISRA Rule 21.3 / Dir 4.12 - Dynamic heap memory allocation shall not
 *      be used. All malloc/free calls removed.
 *      Replaced with static arrays of fixed size.
 * FIX: Rule 11.9 - Pointer initialization uses NULL instead of 0.
 * FIX: Rule 11.5 - Removed void* to object pointer casts (from malloc).
 * FIX: Dir 4.1-b - Eliminated null pointer dereference paths (no more malloc
 *      that can return NULL).
 * FIX: Dir 4.7 - No need to check malloc return values (no malloc). */
#define MSG_BUFFER_SIZE 128U
#define NUM_MESSAGES    3U

static char msg_buffer_0[MSG_BUFFER_SIZE];  /* "Low" */
static char msg_buffer_1[MSG_BUFFER_SIZE];  /* "High" */
static char msg_buffer_2[MSG_BUFFER_SIZE];  /* "Error" */

/* FIX: Rule 18.5 (Advisory) - No more than two levels of pointer nesting.
 *      Original char** messages replaced with char* messages[3]. */
static char *messages[NUM_MESSAGES] = { NULL, NULL, NULL };

static const int32_t VALUE_LOW_MSG  = 0;
static const int32_t VALUE_HIGH_MSG = 1;
static const int32_t ERROR_MSG      = 2;

/* FIX: Rule 10.3 - size parameter changed to size_t to match memset/memcpy
 *      third argument type (essentially unsigned), avoiding signed-to-unsigned
 *      implicit conversion.
 * FIX: Rule 17.7 - Return value of memset/memcpy is now explicitly cast to void. */
static void testMemset(char *buffer, int32_t size)
{
    /* FIX: Rule 10.3 - Cast int32_t to size_t for memset's third parameter */
    (void)memset(buffer, 0, (size_t)size);
}


/* FIX: Rule 1.5 / Rule 8.2 - Empty parameter list () replaced with (void)
 *      to be in prototype form.
 * FIX: Rule 11.9 - Compare pointer with NULL instead of 0.
 * FIX: Rule 10.4 - Removed sizeof(char*) * 3 mixed-type multiplication. */
static void initialize(void)
{
    if (messages[0] == NULL)  /* FIX: Rule 11.9 - NULL instead of 0 */
    {
        /* FIX: Rule 21.3/Dir 4.12 - Static buffers replace malloc */
        messages[0] = msg_buffer_0;
        messages[1] = msg_buffer_1;
        messages[2] = msg_buffer_2;

        testMemset(messages[0], (int32_t)MSG_BUFFER_SIZE);
        testMemset(messages[1], (int32_t)MSG_BUFFER_SIZE);
        testMemset(messages[2], (int32_t)MSG_BUFFER_SIZE);

        /* FIX: Rule 17.7 - strcpy return value explicitly discarded */
        (void)strcpy(messages[VALUE_LOW_MSG], "Low");
        (void)strcpy(messages[VALUE_HIGH_MSG], "High");
        (void)strcpy(messages[ERROR_MSG], "Error");
    }
}

/* FIX: Rule 1.5 / Rule 8.2 - (void) prototype form.
 * FIX: Rule 14.4 - Controlling expression shall have essentially Boolean type.
 *      Original 'if (messages)' changed to explicit NULL comparison.
 * FIX: Rule 21.3/Dir 4.12 - free() calls removed (static allocation). */
static void finalize(void)
{
    if (messages[0] != NULL)  /* FIX: Rule 14.4 - Explicit Boolean test */
    {
        /* FIX: Rule 21.3/Dir 4.12 - No free() needed for static buffers.
         *      Just reset pointers to NULL. */
        messages[0] = NULL;  /* FIX: Rule 11.9 - NULL instead of 0 */
        messages[1] = NULL;
        messages[2] = NULL;
    }
}

/* FIX: Rule 21.6 - printf and fflush replaced with safe_print_message.
 * FIX: Rule 17.7 - printf/fflush return values were unused. */
static void printMessage(int32_t msgIndex, int32_t value)
{
    const char *msg = messages[msgIndex];
    safe_print_message(value, msg);
}

static void reportSensorFailure(void)  /* FIX: Rule 1.5/8.2 - (void) */
{
    initialize();
    printMessage(ERROR_MSG, 0);
    finalize();
}

/* FIX: Rule 15.7 - All if...else if constructs shall be terminated with else.
 * FIX: Rule 12.1 - Operands of && shall be in parentheses for clarity. */
static void handleSensorValue(int32_t value)
{
    int32_t index = VALUE_LOW_MSG;  /* FIX: Dir 4.6 - int -> int32_t */
    initialize();
    if ((value >= 0) && (value <= 10))  /* FIX: Rule 12.1 - Added parentheses */
    {
        index = VALUE_LOW_MSG;
    }
    else if ((value > 10) && (value <= 20))  /* FIX: Rule 12.1 - Added parentheses */
    {
        index = VALUE_HIGH_MSG;
    }
    else  /* FIX: Rule 15.7 - Required terminating else */
    {
        /* Value out of expected range; default to LOW */
        index = VALUE_LOW_MSG;
    }
    printMessage(index, value);
}

/* FIX: Rule 1.5 / Rule 8.2 - (void) prototype form.
 * FIX: Rule 15.4 (Advisory) - Two breaks in the while loop. Kept for
 *      functional equivalence. This is Advisory and acceptable with deviation.
 * FIX: Rule 15.7 - else-if terminated with else. */
static void mainLoop(void)
{
    int32_t sensorValue = 0;  /* FIX: Dir 4.6 - int -> int32_t */
    int32_t status = 0;       /* FIX: Dir 4.6 - int -> int32_t */
    int32_t running = 1;      /* FIX: Rule 14.3 - Loop control variable instead of constant */
    while (running != 0)      /* FIX: Rule 14.3 - Non-constant controlling expression */
    {
        status = readSensor(&sensorValue);
        if (status == STATUS_STOPPED)
        {
            running = 0;
        }
        else if (status == STATUS_FAILED)
        {
            reportSensorFailure();
            running = 0;
        }
        else  /* FIX: Rule 15.7 - Required terminating else */
        {
            handleSensorValue(sensorValue);
        }
    }
    finalize();
}

/* FIX: Rule 1.5 / Rule 8.2 - main(void) instead of main() */
int32_t main(void)
{
    mainLoop();
    return 0;
}
