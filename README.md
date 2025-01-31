# point_cloud_reg

A small project for 3D point cloud **segmentation**, **measurement**, and **validation**.  

## Overview

- **block_segmentation.py**:  
  Removes the max plane (e.g., a table) from point clouds and clusters the remaining objects. Outputs each cluster to disk.  

- **measure_block.py**:  
  Takes a segmented block cluster, removes the top plane, and measures the distance between opposing faces.  

- **validation_registration.py**:  
  Computes various error metrics (RMSE, Chamfer, Hausdorff, etc.) between registered and ground truth point clouds.  

More features and documentation will be added in the future.

## Installation

1. Create and activate a Python 3.8+ virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```
2. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```