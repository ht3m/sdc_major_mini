import json
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import groupby, count

class TrajectoryStabilizer:
    def __init__(self):
        # === Configuration ===
        # Target Joint to fix (0-based index): 1 = Joint 2
        self.TARGET_JOINT = 0  
        
        # Detection parameters
        self.CHECK_WINDOW = 5        # Check window size (frames)
        self.DETECT_THRESHOLD = 3.0  # Detection threshold (degrees)
        self.RECOVERY_THRESHOLD = 1.0 # Recovery threshold (degrees)

    def fix_joint_jumps(self, joints):
        """
        Apply 'Sample and Hold' strategy to remove artifacts from a specific joint.
        """
        print(f"🔧 Stabilizing Joint {self.TARGET_JOINT + 1}...")
        
        # Extract data for the target joint (convert to degrees for processing)
        original_rad = joints[:, self.TARGET_JOINT].copy()
        fixed_deg = np.degrees(original_rad)
        
        fix_count = 0
        modified_indices = []
        
        i = self.CHECK_WINDOW
        total_len = len(fixed_deg)
        
        while i < total_len - self.CHECK_WINDOW:
            # A. Get current window
            current_window = fixed_deg[i : i + self.CHECK_WINDOW]
            
            # B. Check for jump (Peak-to-Peak)
            ptp = np.ptp(current_window)
            
            if ptp > self.DETECT_THRESHOLD:
                # === Jump Detected ===
                
                # 1. Sample: Calculate stable mean before the jump
                prev_window = fixed_deg[i - self.CHECK_WINDOW : i]
                stable_target = np.mean(prev_window)
                
                # 2. Search Recovery: Look ahead for when it stabilizes
                recovery_idx = -1
                for k in range(i + 1, total_len):
                    diff = abs(fixed_deg[k] - stable_target)
                    if diff <= self.RECOVERY_THRESHOLD:
                        recovery_idx = k
                        break
                
                # 3. Hold & Fix
                if recovery_idx != -1:
                    # Flatten the jump area to the stable value
                    fixed_deg[i : recovery_idx] = stable_target
                    modified_indices.extend(range(i, recovery_idx))
                    fix_count += 1
                    i = recovery_idx 
                else:
                    # If no recovery found, hold until end
                    fixed_deg[i:] = stable_target
                    modified_indices.extend(range(i, total_len))
                    print(f"⚠️ Warning: Could not recover from frame {i}, holding value until end.")
                    break
            else:
                i += 1

        # Apply changes back to the main array (convert back to radians)
        joints[:, self.TARGET_JOINT] = np.radians(fixed_deg)
        
        print(f"✅ Joint {self.TARGET_JOINT + 1} Stabilization Complete.")
        print(f"   - Glitches fixed: {fix_count}")
        print(f"   - Total frames modified: {len(modified_indices)}")
        
        return joints, modified_indices

    def visualize_changes(self, original_joints, fixed_joints, modified_indices):
        """ Generate a comparison plot """
        if len(modified_indices) == 0:
            return

        print("📊 Generating comparison plot...")
        j_idx = self.TARGET_JOINT
        orig_deg = np.degrees(original_joints[:, j_idx])
        fix_deg = np.degrees(fixed_joints[:, j_idx])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot Original
        ax1.plot(orig_deg, color='#1f77b4', linewidth=1, label='Original')
        ax1.set_title(f"Step 08: Joint {j_idx+1} Raw", fontweight='bold')
        ax1.set_ylabel("Angle (deg)")
        ax1.grid(True, alpha=0.3)
        
        # Mark jumps on original
        if len(modified_indices) > 0:
            jump_y = orig_deg[modified_indices]
            ax1.scatter(modified_indices, jump_y, color='red', s=10, alpha=0.5, label='Jumps Detected')
            ax1.legend()

        # Plot Fixed
        ax2.plot(fix_deg, color='#2ca02c', linewidth=1.5, label='Stabilized')
        ax2.set_title(f"Step 09: Joint {j_idx+1} Stabilized", fontweight='bold')
        ax2.set_ylabel("Angle (deg)")
        ax2.set_xlabel("Frame Index")
        ax2.grid(True, alpha=0.3)

        # Highlight modified areas
        modified_indices = sorted(list(set(modified_indices)))
        def as_range(g):
            l = list(g)
            return l[0], l[-1]
        ranges = [as_range(g) for _, g in groupby(modified_indices, key=lambda n, c=count(): n-next(c))]
        
        for start, end in ranges:
            ax2.axvspan(start, end, color='red', alpha=0.15)

        plt.tight_layout()
        # Save plot to img folder
        plot_path = os.path.join(os.path.dirname(OUTPUT_FILE_FULL), "step09_stabilization_report.png")
        plt.savefig(plot_path)
        print(f"📈 Report saved: {plot_path}")
        # plt.show() # Uncomment if you want to see the window

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
    stabilized_joints, mod_indices = stabilizer.fix_joint_jumps(joints_np)
    
    # === Visualization ===
    stabilizer.visualize_changes(original_joints_np, stabilized_joints, mod_indices)

    # === Save Files ===
    
    # 1. Save Full JSON (for Python/Matlab)
    output_full = {
        "meta": {
            "source": "step09_stabilizer",
            "parent_meta": meta,
            "modifications": f"Fixed J{stabilizer.TARGET_JOINT+1} artifacts",
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