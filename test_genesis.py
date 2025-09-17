#!/usr/bin/env python3
"""
Test two boxes with different collision settings using Genesis framework.
One box has collision disabled (contype=0), the other has collision enabled.
"""

import os
import sys
from pathlib import Path
import genesis as gs
import imageio
import numpy as np

def test_boxes_simple():
    """Test two boxes with different collision properties."""
    
    print("\n" + "="*70)
    print("SIMPLE TWO BOXES TEST using Genesis")
    print("="*70)
    
    # Initialize Genesis
    gs.init(backend=gs.cpu)
    
    # Use the two_boxes.xml file
    xml_file = "two_boxes.xml"
    output_dir = "output"
    
    print(f"XML file: {xml_file}")
    print(f"Output dir: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scene without interactive viewer for headless recording
    scene = gs.Scene(
        show_viewer=False,  # Disable interactive viewer for headless recording
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(0, -3, 1.5),
            camera_lookat=(0, 0, 0.5),
            camera_fov=60,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            ambient_light=(0.3, 0.3, 0.3),
        ),
        sim_options=gs.options.SimOptions(
            dt=0.002,
            gravity=(0.0, 0.0, -9.81),
        ),
    )
    
    # Load the MJCF model
    model = scene.add_entity(
        gs.morphs.MJCF(file=xml_file)
    )
    
    # Add a camera for recording with rendering enabled
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(0, -3, 1.5),
        lookat=(0, 0, 0.5),
        fov=60,
        GUI=False,
    )
    
    # Build the scene
    scene.build()
    
    try:
        print("\nStarting simulation...")
        print("- Red box (left): Collision DISABLED (contype=0)")
        print("- Blue box (right): Collision ENABLED (contype=1)")
        print("- Yellow ball should fall through red box")
        print("- Green ball should land on blue box")
        
        # Run simulation for 5 seconds (shorter duration for testing)
        sim_duration = 5.0
        sim_steps = int(sim_duration / 0.002)
        fps = 30
        frame_interval = int(1.0 / (fps * 0.002))  # Record every N steps for 30 FPS
        
        frames = []
        print(f"\nRecording {sim_duration} seconds at {fps} FPS...")
        print(f"Total steps: {sim_steps}, Frame interval: {frame_interval}")
        
        for i in range(sim_steps):
            scene.step()
            
            # Capture frames at desired FPS interval
            if i % frame_interval == 0:
                # Get the frame from camera
                img = cam.render(rgb=True, depth=False, segmentation=False)
                if img is not None:
                    # Handle tuple output (if it returns rgb and depth)
                    if isinstance(img, tuple):
                        img = img[0]  # Get just the RGB component
                    # Ensure it's a numpy array and convert to uint8
                    img = np.asarray(img)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:  # If normalized 0-1
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                    frames.append(img)
            
            # Print progress every 250 steps
            if i % 250 == 0:
                print(f"  Step {i}/{sim_steps} ({100*i/sim_steps:.1f}%), Frames captured: {len(frames)}")
        
        # Save video using imageio
        if frames:
            video_path = f"{output_dir}/genesis_boxes_test.mp4"
            print(f"\nSaving {len(frames)} frames to video...")
            imageio.mimsave(video_path, frames, fps=fps)
            print("✅ Simulation completed successfully!")
            print(f"📹 Video saved to: {video_path}")
        else:
            print("⚠️ No frames captured!")
            print("❌ Failed to record video")
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        scene.destroy()

if __name__ == "__main__":
    test_boxes_simple()