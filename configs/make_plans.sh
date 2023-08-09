#!/bin/sh

# <compressed> <uncompressed>
../bin/show_chord_plan int8 float16 >plan_int8_float16.txt
../bin/show_chord_plan int8 float32 >plan_int8_float32.txt
../bin/show_chord_plan float16 float16 >plan_float16_float16.txt
../bin/show_chord_plan float16 float32 >plan_float16_float32.txt
../bin/show_chord_plan float32 float32 >plan_float32_float32.txt

