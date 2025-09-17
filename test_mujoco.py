#!/usr/bin/env python3
"""
Test two boxes with different collision settings using MuJoCo framework.
One box has collision disabled (contype=0), the other has collision enabled.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("MuJoCo not installed. Install with: pip install mujoco")
    sys.exit(1)

try:
    import imageio
    import matplotlib.pyplot as plt
except ImportError:
    print("Additional packages needed. Install with: pip install imageio matplotlib")
    sys.exit(1)


def test_boxes_simple():
    """Test two boxes with different collision properties using MuJoCo."""
    
    print("\n" + "="*70)
    print("SIMPLE TWO BOXES TEST using MuJoCo")
    print("="*70)
    
    # Use the test_two_boxes.xml file
    xml_file = "two_boxes.xml"
    output_dir = "output"
    
    print(f"XML file: {xml_file}")
    print(f"Output dir: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    try:
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\nModel loaded successfully!")
    print(f"- Number of geoms: {model.ngeom}")
    print(f"- Number of bodies: {model.nbody}")
    print(f"- Timestep: {model.opt.timestep}")
    
    # Print collision information for each geom
    print("\nGeom collision properties:")
    for i in range(model.ngeom):
        name = model.geom(i).name if model.geom(i).name else f"geom_{i}"
        contype = model.geom_contype[i]
        conaffinity = model.geom_conaffinity[i]
        print(f"  - {name}: contype={contype}, conaffinity={conaffinity}")
    
    # Create renderer for recording
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Simulation parameters
    sim_duration = 5.0  # seconds (same as Genesis test)
    fps = 30  # 30 FPS for smooth video
    frame_skip = int(1.0 / (fps * model.opt.timestep))
    n_frames = int(sim_duration * fps)
    
    frames = []
    
    try:
        print("\nStarting simulation...")
        print("- Red box (left): Collision DISABLED (contype=0)")
        print("- Blue box (right): Collision ENABLED (contype=1)")
        print("- Yellow ball should fall through red box")
        print("- Green ball should land on blue box")
        print("\nRunning simulation and recording frames...")
        print(f"Recording {n_frames} frames at {fps} FPS for {sim_duration} seconds")
        
        # Option: Run headless for consistent video recording
        use_viewer = False  # Set to True if you want interactive viewer
        
        if use_viewer:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                start = time.time()
                frame_count = 0
                
                while viewer.is_running() and time.time() - start < sim_duration:
                    step_start = time.time()
                    
                    # Step the simulation
                    mujoco.mj_step(model, data)
                    
                    # Record frame periodically
                    if data.time * fps >= frame_count:
                        # Update renderer
                        renderer.update_scene(data, camera="main_cam")
                        frame = renderer.render()
                        frames.append(frame)
                        frame_count += 1
                        
                        # Print progress
                        if frame_count % 30 == 0:
                            print(f"  Frame {frame_count}/{n_frames} ({100*frame_count/n_frames:.1f}%)")
                    
                    # Sync with wall-clock time for viewer
                    viewer.sync()
                    
        else:
            # Option 2: Run headless (no viewer)
            frame_count = 0
            total_steps = int(sim_duration / model.opt.timestep)
            
            for step in range(total_steps):
                # Step the simulation
                mujoco.mj_step(model, data)
                
                # Record frame periodically
                if step % frame_skip == 0:
                    # Update renderer
                    renderer.update_scene(data, camera="main_cam")
                    frame = renderer.render()
                    frames.append(frame)
                    frame_count += 1
                    
                    # Print progress
                    if frame_count % 30 == 0:
                        print(f"  Frame {frame_count}/{n_frames} ({100*frame_count/n_frames:.1f}%)")
        
        # Save video
        if frames:
            video_path = f"{output_dir}/mujoco_boxes_test.mp4"
            print(f"\nSaving video with {len(frames)} frames...")
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"📹 Video saved to: {video_path}")
        
        # Save final frame as image
        if frames:
            final_frame_path = f"{output_dir}/mujoco_final_frame.png"
            plt.figure(figsize=(12, 8))
            plt.imshow(frames[-1])
            plt.axis('off')
            plt.title("Final Frame - MuJoCo Simulation")
            plt.tight_layout()
            plt.savefig(final_frame_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📷 Final frame saved to: {final_frame_path}")
        
        # Print final positions of the balls
        print("\nFinal positions:")
        ball1_pos = data.geom_xpos[model.geom("ball1_geom").id]
        ball2_pos = data.geom_xpos[model.geom("ball2_geom").id]
        print(f"  - Yellow ball (left): z = {ball1_pos[2]:.3f} (should be ~0.1, passed through)")
        print(f"  - Green ball (right): z = {ball2_pos[2]:.3f} (should be ~0.9, on top of box)")
        
        # Check if collision behavior is as expected
        if ball1_pos[2] < 0.3:  # Ball went through the red box
            print("✅ Yellow ball correctly passed through red box (no collision)")
        else:
            print("❌ Yellow ball did not pass through red box (unexpected)")
        
        if ball2_pos[2] > 0.7:  # Ball stayed on top of blue box
            print("✅ Green ball correctly landed on blue box (collision enabled)")
        else:
            print("❌ Green ball did not land on blue box (unexpected)")
        
        print("\n✅ Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        renderer.close()
        print("\nCleanup complete.")


if __name__ == "__main__":
    test_boxes_simple()
