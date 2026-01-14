import json
import numpy as np
import os
import matplotlib.pyplot as plt

class TrajectoryStabilizer:
    def __init__(self):
        # === Configuration ===
        # Target Joint to fix (0-based index): 0 = Joint 1
        self.TARGET_JOINT = 0  
        
        # Detection parameters
        self.CHECK_WINDOW = 5        # Check window size (frames)
        self.DETECT_THRESHOLD = 5.0  # Detection threshold (degrees)
        self.RECOVERY_THRESHOLD = 8.0 # Recovery threshold (degrees)

    def fix_joint_jumps(self, joints):
        """
        Strategy: Preserve Length + Local Interpolation
        1. Detect jump in TARGET_JOINT.
        2. Find recovery point.
        3. Linearly interpolate TARGET_JOINT values between start and recovery.
        4. Keep other joints UNCHANGED.
        """
        print(f"🔧 Stabilizing Joint {self.TARGET_JOINT + 1}...")
        
        # Make a copy to modify
        fixed_joints = joints.copy()
        n_frames, n_joints = fixed_joints.shape
        
        # Convert to degrees for easier threshold checking
        target_vals_deg = np.degrees(fixed_joints[:, self.TARGET_JOINT])
        
        i = self.CHECK_WINDOW
        fix_count = 0
        modified_ranges = [] # Store (start, end) for visualization
        
        while i < n_frames - self.CHECK_WINDOW:
            # Check window for Jumps
            window_vals = target_vals_deg[i : i + self.CHECK_WINDOW]
            ptp = np.ptp(window_vals)
            
            if ptp > self.DETECT_THRESHOLD:
                # === Jump Detected ===
                # Assume i-1 is the last stable point
                start_idx = i - 1
                start_val = fixed_joints[start_idx, self.TARGET_JOINT]
                start_val_deg = target_vals_deg[start_idx]
                
                # Search for recovery point
                recovery_idx = -1
                for k in range(i + 1, n_frames):
                    val_k_deg = target_vals_deg[k]
                    # Check if value returns to stable range
                    if abs(val_k_deg - start_val_deg) <= self.RECOVERY_THRESHOLD:
                        recovery_idx = k
                        break
                
                if recovery_idx != -1:
                    # === Recovery Found ===
                    end_val = fixed_joints[recovery_idx, self.TARGET_JOINT]
                    
                    # Calculate number of frames to interpolate
                    # range is from start_idx+1 to recovery_idx-1
                    # Total steps including start and end
                    steps = recovery_idx - start_idx
                    
                    # Generate interpolated values for TARGET_JOINT
                    # np.linspace includes start and end, we extract the middle part
                    interp_values = np.linspace(start_val, end_val, num=steps+1)
                    
                    # Apply interpolation (excluding start, including end to ensure continuity)
                    # Actually better to exclude both if we want to be strict, 
                    # but assigning from i to recovery_idx (exclusive) matches the gap.
                    
                    # Indices to replace: i ... recovery_idx-1
                    # interp_values[1:-1] are the points strictly between start and end
                    
                    # Example: start=10, jump at 11, 12, recov at 13.
                    # i=11. start_idx=10. recovery_idx=13.
                    # steps = 3.
                    # linspace(10, 13, num=4) -> [v10, v11', v12', v13]
                    # we want to replace 11 and 12.
                    
                    gap_values = interp_values[1:-1]
                    
                    # Check length match
                    expected_len = recovery_idx - i # (13 - 11 = 2)
                    if len(gap_values) != expected_len:
                        # Fallback for edge cases (should not happen with linspace logic above)
                        print(f"⚠️ Interpolation length mismatch at {i}")
                    else:
                        fixed_joints[i:recovery_idx, self.TARGET_JOINT] = gap_values
                        modified_ranges.append((i, recovery_idx))
                        fix_count += 1
                        print(f"   -> Fixed range [{i}:{recovery_idx}]. Interpolated {len(gap_values)} frames.")

                    # Move index forward
                    i = recovery_idx
                else:
                    # No recovery found, skip frame
                    i += 1
            else:
                i += 1
                
        print(f"✅ Joint {self.TARGET_JOINT + 1} Stabilization Complete.")
        print(f"   - Glitches fixed: {fix_count}")
        
        return fixed_joints, modified_ranges

    def visualize_changes(self, original_joints, fixed_joints, modified_ranges):
        """ Generate a comparison plot """
        print("📊 Generating comparison plot...")
        j_idx = self.TARGET_JOINT
        orig_deg = np.degrees(original_joints[:, j_idx])
        fix_deg = np.degrees(fixed_joints[:, j_idx])

        # sharex=True is safe now because lengths are preserved
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot Original
        ax1.plot(orig_deg, color='#1f77b4', linewidth=1, label='Original')
        ax1.set_title(f"Step 08: Joint {j_idx+1} Raw", fontweight='bold')
        ax1.set_ylabel("Angle (deg)")
        ax1.grid(True, alpha=0.3)
        
        # Plot Fixed
        ax2.plot(fix_deg, color='#2ca02c', linewidth=1.5, label='Stabilized')
        ax2.set_title(f"Step 09: Joint {j_idx+1} Stabilized (Length Preserved)", fontweight='bold')
        ax2.set_ylabel("Angle (deg)")
        ax2.set_xlabel("Frame Index")
        ax2.grid(True, alpha=0.3)
        
        # Highlight modified areas
        for start, end in modified_ranges:
            ax1.axvspan(start, end, color='red', alpha=0.15, label='Jump' if start == modified_ranges[0][0] else "")
            ax2.axvspan(start, end, color='green', alpha=0.15, label='Interpolated' if start == modified_ranges[0][0] else "")

        plt.tight_layout()
        # Save plot to img folder
        plot_path = os.path.join(os.path.dirname(OUTPUT_FILE_FULL), "step09_stabilization_report.png")
        plt.savefig(plot_path)
        print(f"📈 Report saved: {plot_path}")
        # plt.show() 

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # === I/O Paths ===
    INPUT_FILE = os.path.join(img_dir, "step08_optimized_trajectory.json")
    
    # Standard format (with meta)
    OUTPUT_FILE_FULL = os.path.join(img_dir, "step09_stable_traj.json")
    # Rust format (for move_traj_from_file)
    OUTPUT_FILE_RUST = os.path.join(img_dir, "step09_stable_traj_rust.json")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        exit()

    print(f"📂 Loading: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # Handle data format
    if isinstance(data, dict) and "joints" in data:
        joints_list = data["joints"]
        meta = data.get("meta", {})
    else:
        print("❌ Invalid JSON format (expected 'joints' key)")
        exit()

    joints_np = np.array(joints_list)
    original_joints_np = joints_np.copy() # Keep a copy for visualization

    # === Process ===
    stabilizer = TrajectoryStabilizer()
    stabilized_joints, modified_ranges = stabilizer.fix_joint_jumps(joints_np)
    
    # === Visualization ===
    stabilizer.visualize_changes(original_joints_np, stabilized_joints, modified_ranges)

    # === Save Files ===
    
    # 1. Save Full JSON (for Python/Matlab)
    output_full = {
        "meta": {
            "source": "step09_stabilizer",
            "parent_meta": meta,
            "modifications": f"Fixed J{stabilizer.TARGET_JOINT+1} artifacts (In-place Interpolation)",
            "count": len(stabilized_joints)
        },
        "joints": stabilized_joints.tolist()
    }
    
    with open(OUTPUT_FILE_FULL, 'w') as f:
        json.dump(output_full, f, indent=2)
    print(f"💾 Saved Full JSON: {OUTPUT_FILE_FULL}")

    # 2. Save Rust JSON (Enum format)
    # [{"Joint": [...]}, {"Joint": [...]}]
    rust_data = [{"Joint": frame} for frame in stabilized_joints.tolist()]
    
    with open(OUTPUT_FILE_RUST, 'w') as f:
        json.dump(rust_data, f, indent=2)
    print(f"🦀 Saved Rust JSON: {OUTPUT_FILE_RUST}")