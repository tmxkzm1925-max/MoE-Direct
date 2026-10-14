// Package selftest holds the test of the Go self-test entry MoE-Direct-launcher-selftest.ps1
// at the module root (K194, version 8): TestS8HX12 starts real processes to show the entry's
// duplicate guard follows the rule of KNOWN row 194. The entry runs go vet and go test over
// this module and ends with the receipt line; it is not shipped. Plain go test is unguarded.
package selftest
