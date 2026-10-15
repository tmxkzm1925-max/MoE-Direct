#pragma once
// Suite entry points that live in their own translation units.
// One line per suite; the explicit registry in main() stays the ordering authority.
void test_memplan_s1();   // P41 - suite-memplan-s2.cpp
void test_harness_teardown();   // HT - selftest-harness.cpp
