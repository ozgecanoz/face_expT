#!/usr/bin/env python3
import h5py

# Open the H5 file
with h5py.File('/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_81_1220_10_faces_4_43.h5', 'r') as f:
    print("Root keys:", list(f.keys()))
    
    if 'faces' in f:
        print("\nFaces group keys:", list(f['faces'].keys()))
        if 'frame_000' in f['faces']:
            frame = f['faces/frame_000']
            print(f"Frame 000 shape: {frame.shape}")
            print(f"Frame 000 dtype: {frame.dtype}")
            print(f"Frame 000 min/max: {frame[:].min()}, {frame[:].max()}")
    
    if 'metadata' in f:
        print("\nMetadata group keys:", list(f['metadata'].keys()))
        for key in f['metadata'].keys():
            item = f[f'metadata/{key}']
            if hasattr(item, 'shape'):
                print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  {key}: {item}")
