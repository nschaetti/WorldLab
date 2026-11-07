"""
Test script to verify that the imports work correctly after refactoring.
"""

try:
    from worldlab.nn.board import (
        GPT, 
        GPTConfig, 
        LayerNorm, 
        CausalSelfAttention, 
        Block, 
        MLP
    )
    print("All imports successful!")
    print("Imported classes:")
    print(f"- GPT: {GPT}")
    print(f"- GPTConfig: {GPTConfig}")
    print(f"- LayerNorm: {LayerNorm}")
    print(f"- CausalSelfAttention: {CausalSelfAttention}")
    print(f"- Block: {Block}")
    print(f"- MLP: {MLP}")
except ImportError as e:
    print(f"Import error: {e}")