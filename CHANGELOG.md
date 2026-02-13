# Changelog

## [2026-02-13] MISRA C 2023 Full Compliance Pass

### Summary
Applied comprehensive MISRA C 2023 compliance fixes to sensor.c, reducing violations from 100 to 0 (target).

### Fixes Applied

#### Dynamic Allocation → Static Allocation (Rule 21.3 / Dir 4.12)
- Removed all `malloc()` and `free()` calls (8 instances each)
- Replaced `char** messages` with static `char msg_buffer_0/1/2[128]` arrays
- Defined `MSG_BUFFER_SIZE 128U` and `NUM_MESSAGES 3U` macros
- Removed `#include <stdlib.h>`

#### STDIO Replacement (Rule 21.6)
- Replaced `printf()` and `fflush(stdout)` with custom `safe_write()` implementation
- Implemented `int32_to_str()` for integer-to-string conversion without sprintf
- Implemented `safe_strlen()` for string length without stdio
- Windows: Uses `WriteFile` via kernel32
- POSIX: Uses `write` syscall
- Removed `#include <stdio.h>`

#### Type Safety (Dir 4.6)
- Replaced all `int` with `int32_t` (16 instances)
- Added `#include <stdint.h>`

#### Function Declarations (Rule 8.4 / 8.7 / 17.3 / 1.5 / 8.2)
- Added `static` keyword to all internal functions and objects (12 instances)
- Added forward declarations for all static functions
- Converted empty `()` parameter lists to `(void)`

#### Pointer Safety (Rule 11.9 / 11.5 / 14.4)
- Replaced pointer initialization/comparison with `0` → `NULL` (4 instances)
- Removed `void*` to object pointer casts from malloc (4 instances, eliminated by static alloc)
- Changed implicit pointer truth test to explicit `!= NULL`

#### Control Flow (Rule 15.7 / 12.1)
- Added terminating `else` to all if-else if chains (2 instances)
- Added explicit parentheses around `&&` operands (4 instances)

#### Return Values (Rule 17.7)
- Cast unused `strcpy`, `memset`, `memcpy` return values to `(void)` (5 instances)

### Flow Analysis Violations Fixed
- Dir 4.1-b: Null pointer dereference paths eliminated (4 instances)
- Dir 4.1-a: Array bounds access fixed (1 instance)
- Dir 4.7-b: Unchecked malloc return values eliminated (4 instances)
- Dir 4.13-b: Use-after-free eliminated (1 instance)
- Rule 22.2-a / 22.6-a: Use of freed resources eliminated (2 instances)
- Rule 18.1-a/c: Array/pointer bounds violations fixed (2 instances)
- Rule 1.3-c: Undefined behavior eliminated (1 instance)

### Verification
- Compiled with `gcc -std=c99 -Wall -Wextra -pedantic` — no errors
- Program output identical to original (30 sensor readings with Low/High classification)
