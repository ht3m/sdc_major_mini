import json
import numpy as np
import os
import matplotlib.pyplot as plt

class TrajectoryStabilizer:
    def __init__(self):
        # === Configuration ===
        # Detection parameters
        self.CHECK_WINDOW = 5        # Check window size (frames)
        self.DETECT_THRESHOLD = 5.0  # Detection threshold (degrees)
        self.RECOVERY_THRESHOLD = 8.0 # Recovery threshold (degrees)

    def stabilize_all_joints(self, joints):
        """
        Iterate through all joints (0-5) and fix jumps while preserving trajectory length.
        Returns:
            fixed_joints: The fully corrected numpy array.
            modification_report: Dict { joint_index: [(start, end), (start, end)...] }
        """
        # Make a copy to modify
        fixed_joints = joints.copy()
        n_frames, n_joints = fixed_joints.shape
        
        # Dictionary to store modification ranges for each joint
        # Key: joint index, Value: list of (start, end) tuples
        modification_report = {}

        print("🚀 Starting Stabilization for All Joints (1-6)...")
        print("-" * 50)

        # Loop through every joint column
        for j_idx in range(n_joints):
            print(f"🔍 Checking Joint {j_idx + 1}...")
            
            # Extract column for specific joint
            target_vals_deg = np.degrees(fixed_joints[:, j_idx])
            
            i = self.CHECK_WINDOW
            fix_count = 0
            current_joint_changes = [] # Store (start, end) for this joint
            
            while i < n_frames - self.CHECK_WINDOW:
                # Check window for Jumps
                window_vals = target_vals_deg[i : i + self.CHECK_WINDOW]
                ptp = np.ptp(window_vals)
                
                if ptp > self.DETECT_THRESHOLD:
                    # === Jump Detected ===
                    start_idx = i - 1
                    start_val = fixed_joints[start_idx, j_idx]
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
                        end_val = fixed_joints[recovery_idx, j_idx]
                        steps = recovery_idx - start_idx
                        
                        # Generate interpolated values
                        interp_values = np.linspace(start_val, end_val, num=steps+1)
                        gap_values = interp_values[1:-1]
                        
                        expected_len = recovery_idx - i
                        
                        if len(gap_values) == expected_len:
                            # Apply fix to the main array
                            fixed_joints[i:recovery_idx, j_idx] = gap_values
                            
                            # Also update our local view (deg) so subsequent checks in this loop see the fix
                            target_vals_deg[i:recovery_idx] = np.degrees(gap_values)
                            
                            current_joint_changes.append((i, recovery_idx))
                            fix_count += 1
                            print(f"   -> 🛠️ Fixed J{j_idx+1} range [{i}:{recovery_idx}].")
                        else:
                            print(f"   ⚠️ Interpolation length mismatch at {i}")

                        # Move index forward
                        i = recovery_idx
                    else:
                        # No recovery found, skip frame
                        i += 1
                else:
                    i += 1
            
            # If we made changes to this joint, record them
            if fix_count > 0:
                modification_report[j_idx] = current_joint_changes
                print(f"   ✅ Joint {j_idx + 1}: Fixed {fix_count} glitches.")
            else:
                print(f"   🆗 Joint {j_idx + 1}: Clean.")
            
            print("-" * 30)
            
        return fixed_joints, modification_report

    def visualize_changes(self, original_joints, fixed_joints, modification_report, output_dir):
        """ Generate comparison plots ONLY for modified joints """
        if not modification_report:
            print("✨ No changes made to any joints. Skipping visualization.")
            return

        print(f"📊 Generating comparison plots for {len(modification_report)} modified joints...")
        
        for j_idx, ranges in modification_report.items():
            orig_deg = np.degrees(original_joints[:, j_idx])
            fix_deg = np.degrees(fixed_joints[:, j_idx])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot Original
            ax1.plot(orig_deg, color='#1f77b4', linewidth=1, label='Original')
            ax1.set_title(f"Joint {j_idx+1} Raw (Glitches Detected)", fontweight='bold')
            ax1.set_ylabel("Angle (deg)")
            ax1.grid(True, alpha=0.3)
            
            # Plot Fixed
            ax2.plot(fix_deg, color='#2ca02c', linewidth=1.5, label='Stabilized')
            ax2.set_title(f"Joint {j_idx+1} Stabilized", fontweight='bold')
            ax2.set_ylabel("Angle (deg)")
            ax2.set_xlabel("Frame Index")
            ax2.grid(True, alpha=0.3)
            
            # Highlight modified areas
            for start, end in ranges:
                ax1.axvspan(start, end, color='red', alpha=0.15, label='Jump' if start == ranges[0][0] else "")
                ax2.axvspan(start, end, color='green', alpha=0.15, label='Interpolated' if start == ranges[0][0] else "")

            # Add legend to first patch only
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')

            plt.tight_layout()
            
            # Save individual plot for this joint
            filename = f"step09_report_joint_{j_idx+1}.png"
            plot_path = os.path.join(output_dir, filename)
            plt.savefig(plot_path)
            print(f"   📈 Saved report: {filename}")
            plt.close(fig) # Close memory

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # === I/O Paths ===
    INPUT_FILE = os.path.join(img_dir, "step08_optimized_trajectory.json")
    OUTPUT_FILE_FULL = os.path.join(img_dir, "step09_stable_traj.json")
    OUTPUT_FILE_RUST = os.path.join(img_dir, "step09_stable_traj_rust.json")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        exit()

    print(f"📂 Loading: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
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
    
    # New method call
    stabilized_joints, modification_report = stabilizer.stabilize_all_joints(joints_np)
    
    # === Visualization ===
    # Pass output directory explicitly
    stabilizer.visualize_changes(original_joints_np, stabilized_joints, modification_report, img_dir)

    # === Save Files ===
    
    # Generate description string
    if modification_report:
        mod_desc = "Fixed: " + ", ".join([f"J{j+1}" for j in modification_report.keys()])
    else:
        mod_desc = "No fixes needed"

    # 1. Save Full JSON
    output_full = {
        "meta": {
            "source": "step09_stabilizer",
            "parent_meta": meta,
            "modifications": mod_desc,
            "count": len(stabilized_joints)
        },
        "joints": stabilized_joints.tolist()
    }
    
    with open(OUTPUT_FILE_FULL, 'w') as f:
        json.dump(output_full, f, indent=2)
    print(f"💾 Saved Full JSON: {OUTPUT_FILE_FULL}")

    # 2. Save Rust JSON
    rust_data = [{"Joint": frame} for frame in stabilized_joints.tolist()]
    
    with open(OUTPUT_FILE_RUST, 'w') as f:
        json.dump(rust_data, f, indent=2)
    print(f"🦀 Saved Rust JSON: {OUTPUT_FILE_RUST}")