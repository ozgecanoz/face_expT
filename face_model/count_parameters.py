#!/usr/bin/env python3
"""
Count parameters in Expression Transformer, Face Reconstruction Model, and Face ID Model
"""

import torch
from models.expression_transformer import ExpressionTransformer
from models.face_reconstruction_model import FaceReconstructionModel
from models.face_id_model import FaceIDModel

def count_parameters(model, model_name):
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name}:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params

def main():
    print("Parameter Count for Components C, D, and E")
    print("=" * 50)
    
    # Component C: Expression Transformer
    expression_transformer = ExpressionTransformer()
    expr_total, expr_trainable = count_parameters(expression_transformer, "Expression Transformer (Component C)")
    
    # Component D: Face ID Model
    face_id_model = FaceIDModel()
    face_id_total, face_id_trainable = count_parameters(face_id_model, "Face ID Model (Component D)")
    
    # Component E: Face Reconstruction Model
    reconstruction_model = FaceReconstructionModel()
    recon_total, recon_trainable = count_parameters(reconstruction_model, "Face Reconstruction Model (Component E)")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Expression Transformer: {expr_total:,} total, {expr_trainable:,} trainable")
    print(f"  Face ID Model: {face_id_total:,} total, {face_id_trainable:,} trainable")
    print(f"  Face Reconstruction Model: {recon_total:,} total, {recon_trainable:,} trainable")
    print(f"  Combined trainable parameters: {expr_trainable + face_id_trainable + recon_trainable:,}")
    
    # Memory usage estimation (rough)
    total_trainable = expr_trainable + face_id_trainable + recon_trainable
    memory_mb = (total_trainable * 4) / (1024 * 1024)  # Assuming float32
    print(f"  Estimated memory for parameters: {memory_mb:.1f} MB")

if __name__ == "__main__":
    main() 