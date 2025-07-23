#!/usr/bin/env python3
"""
Simple test file for debugging
"""

def test_function():
    x = 10
    y = 20
    z = x + y
    print(f"Result: {z}")
    return z

if __name__ == "__main__":
    print("Starting debug test...")
    result = test_function()
    print(f"Final result: {result}") 