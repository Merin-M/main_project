#!/usr/bin/env python
"""
Create visualizations for the best inference results.
Reads metrics from metric_all file, selects top results, and generates comparison images.
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path for imports  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from package_core.generic_train_test import *
from model_SelfRSSR import *
from dataloader import *

def parse_metrics_file(metrics_file):
    """Parse metric_all file and return sorted results by PSNR."""
    results = []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    frame_idx = int(parts[0])
                    psnr = float(parts[1])
                    results.append({'frame_idx': frame_idx, 'psnr': psnr, 'line': line})
                except ValueError:
                    continue
    
    # Sort by PSNR (highest first)
    results.sort(key=lambda x: x['psnr'], reverse=True)
    return results

def create_comparison_image(rs_frame1, rs_frame2, pred_gs, gt_gs, psnr, output_path):
    """Create a 4-panel comparison image."""
    # Ensure all images are same size
    h, w = pred_gs.shape[:2]
    
    # Resize all to same dimensions if needed
    rs_frame1 = cv2.resize(rs_frame1, (w, h))
    rs_frame2 = cv2.resize(rs_frame2, (w, h))
    gt_gs = cv2.resize(gt_gs, (w, h))
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    rs_frame1 = rs_frame1.copy()
    rs_frame2 = rs_frame2.copy()
    pred_gs = pred_gs.copy()
    gt_gs = gt_gs.copy()
    
    cv2.putText(rs_frame1, 'RS Frame (t=0)', (10, 30), font, font_scale, color, thickness)
    cv2.putText(rs_frame2, 'RS Frame (t=1)', (10, 30), font, font_scale, color, thickness)
    cv2.putText(pred_gs, f'Predicted GS (PSNR: {psnr:.2f}dB)', (10, 30), font, font_scale, color, thickness)
    cv2.putText(gt_gs, 'Ground Truth GS', (10, 30), font, font_scale, color, thickness)
    
    # Create 2x2 grid layout (better aspect ratio than horizontal strip)
    top_row = np.hstack([rs_frame1, rs_frame2])
    bottom_row = np.hstack([pred_gs, gt_gs])
    comparison = np.vstack([top_row, bottom_row])
    
    # Save
    cv2.imwrite(str(output_path), comparison)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file', type=str, required=True, help='Path to metric_all file')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of test dataset')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing pretrained models')
    parser.add_argument('--output_dir', type=str, default='../best_visualizations', help='Output directory')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top results to visualize')
    parser.add_argument('--img_H', type=int, default=480)
    parser.add_argument('--gamma', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing metrics from: {args.metrics_file}")
    results = parse_metrics_file(args.metrics_file)
    
    print(f"\nFound {len(results)} results")
    print(f"Top PSNR: {results[0]['psnr']:.2f} dB")
    print(f"Selecting top {args.top_n} results for visualization\n")
    
    # Get top N results
    top_results = results[:args.top_n]
    
    # Load model
    print("Loading pretrained model...")
    
    # Create minimal opts for model
    class SimpleOpts:
        def __init__(self):
            self.model_label = 'pre'
            self.log_dir = args.model_dir
            self.img_H = args.img_H
            self.gamma = args.gamma
            self.is_training = False
            self.continue_train = True
            self.test_pretrained_VFI = False
            self.load_gt_flow = False
            self.load_1st_GS = 1
            self.batch_sz = 1
            self.cH = 0
    
    opts = SimpleOpts()
    
    # Initialize model
    model = Model(opts)
    # Model is already in eval mode from loading pretrained weights
    
    print("Model loaded successfully!\n")
    
    # Get list of sequences
    sequences = sorted([d for d in Path(args.dataset_root).iterdir() if d.is_dir()])
    
    print(f"Processing top {len(top_results)} results...")
    
    for idx, result in enumerate(top_results):
        frame_idx = result['frame_idx']
        psnr = result['psnr']
        
        # Determine which sequence and frame this corresponds to
        # This is approximate - we'll process frames sequentially
        current_frame = 0
        target_seq = None
        target_frame_in_seq = None
        
        for seq_dir in sequences:
            rs_files = sorted(seq_dir.glob('*_rolling.png'))
            n_frames = len(rs_files) - 1  # Number of GS frames we can generate
            
            if current_frame + n_frames > frame_idx:
                target_seq = seq_dir
                target_frame_in_seq = frame_idx - current_frame
                break
            current_frame += n_frames
        
        if target_seq is None:
            print(f"Warning: Could not find sequence for frame {frame_idx}")
            continue
        
        seq_name = target_seq.name
        print(f"{idx+1}/{len(top_results)}: {seq_name}, frame {target_frame_in_seq}, PSNR {psnr:.2f} dB")
        
        # Load RS frames
        rs_files = sorted(target_seq.glob('*_rolling.png'))
        if target_frame_in_seq >= len(rs_files) - 1:
            continue
            
        rs_path1 = rs_files[target_frame_in_seq]
        rs_path2 = rs_files[target_frame_in_seq + 1]
        
        # Load ground truth GS
        gs_files = sorted(target_seq.glob('*_global_middle.png'))
        if target_frame_in_seq >= len(gs_files):
            # Try first GS
            gs_files = sorted(target_seq.glob('*_global_first.png'))
        
        if target_frame_in_seq >= len(gs_files):
            print(f"  Warning: No ground truth GS found")
            continue
            
        gt_gs_path = gs_files[target_frame_in_seq]
        
        # Read images
        rs1 = cv2.imread(str(rs_path1))
        rs2 = cv2.imread(str(rs_path2))
        gt_gs = cv2.imread(str(gt_gs_path))
        
        if rs1 is None or rs2 is None or gt_gs is None:
            print(f"  Warning: Failed to load images")
            continue
        
        # Convert to torch tensors
        rs1_tensor = torch.from_numpy(rs1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        rs2_tensor = torch.from_numpy(rs2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Concatenate RS frames
        im_rs = torch.cat([rs1_tensor, rs2_tensor], dim=1).cuda()
        im_gs = torch.from_numpy(gt_gs).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        
        # Set model input
        model.set_input([im_rs, im_gs, None, 0])
        
        # Run inference
        with torch.no_grad():
            pred_gs_tensor = model.GS_syn(1.0 - opts.gamma/2.0, opts.gamma)
        
        # Convert back to numpy
        pred_gs = (pred_gs_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
       
        # Create visualization
        output_path = output_dir / f"rank_{idx+1:02d}_PSNR_{psnr:.2f}dB_{seq_name}_frame_{target_frame_in_seq}.png"
        create_comparison_image(rs1, rs2, pred_gs, gt_gs, psnr, output_path)
    
    print(f"\nDone! Visualizations saved to: {output_dir}")
    print(f"Generated {len(top_results)} comparison images")

if __name__ == '__main__':
    main()
