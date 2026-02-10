"""
Simple visualization showing the perpendicular vector direction change.
"""

import matplotlib.pyplot as plt
import numpy as np

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Horizontal beam example
x1, y1 = 0, 0
x2, y2 = 1, 0

# Draw beam
for ax in [ax1, ax2]:
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3, label='Beam')
    ax.arrow(x1+0.5, y1, 0.1, 0, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    ax.text(x1+0.5, y1-0.15, 'Element direction →', ha='center', fontsize=10)
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

# OLD perpendicular direction: [-dy, dx] = [0, 1] (UP)
dx_elem = x2 - x1
dy_elem = y2 - y1
norm = np.sqrt(dx_elem**2 + dy_elem**2)
dx_elem /= norm
dy_elem /= norm

perp_old_x = -dy_elem
perp_old_y = dx_elem

# Draw perpendicular vector (OLD)
ax1.arrow(0.5, 0, perp_old_x * 0.3, perp_old_y * 0.3, 
          head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
ax1.text(0.5, 0.35, 'OLD: perp = [-dy, dx] = [0, 1]\nPoints UP', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax1.set_title('OLD Implementation\n(Before Fix)', fontsize=12, fontweight='bold')

# NEW perpendicular direction: [dy, -dx] = [0, -1] (DOWN)
perp_new_x = dy_elem
perp_new_y = -dx_elem

# Draw perpendicular vector (NEW)
ax2.arrow(0.5, 0, perp_new_x * 0.3, perp_new_y * 0.3,
          head_width=0.05, head_length=0.05, fc='green', ec='green', linewidth=2)
ax2.text(0.5, -0.35, 'NEW: perp = [dy, -dx] = [0, -1]\nPoints DOWN', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax2.set_title('NEW Implementation\n(After Fix)', fontsize=12, fontweight='bold')

plt.suptitle('Perpendicular Vector Direction Change\n(Horizontal Beam Example)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/perpendicular_vector_comparison.png', dpi=150, bbox_inches='tight')
print("Saved diagram to: /tmp/perpendicular_vector_comparison.png")

# Create second figure showing diagonal beam
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

# Diagonal beam example (45 degrees)
x1, y1 = 0, 0
x2, y2 = 1, 1

# Draw beam
for ax in [ax3, ax4]:
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=3, label='Beam')
    ax.arrow(x1+0.35, y1+0.35, 0.1, 0.1, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    ax.text(x1+0.3, y1+0.15, 'Element\ndirection', ha='center', fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

# Perpendicular vectors for diagonal beam
dx_elem = x2 - x1
dy_elem = y2 - y1
norm = np.sqrt(dx_elem**2 + dy_elem**2)
dx_elem /= norm
dy_elem /= norm

# OLD: [-dy, dx] = [-0.707, 0.707] (upper-left perpendicular)
perp_old_x = -dy_elem
perp_old_y = dx_elem

ax3.arrow(0.5, 0.5, perp_old_x * 0.4, perp_old_y * 0.4,
          head_width=0.05, head_length=0.05, fc='red', ec='red', linewidth=2)
ax3.text(0.25, 0.95, 'OLD: CCW 90°\nUpper-left', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax3.set_title('OLD Implementation\n(Before Fix)', fontsize=12, fontweight='bold')

# NEW: [dy, -dx] = [0.707, -0.707] (lower-right perpendicular)
perp_new_x = dy_elem
perp_new_y = -dx_elem

ax4.arrow(0.5, 0.5, perp_new_x * 0.4, perp_new_y * 0.4,
          head_width=0.05, head_length=0.05, fc='green', ec='green', linewidth=2)
ax4.text(0.95, 0.25, 'NEW: CW 90°\nLower-right', 
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax4.set_title('NEW Implementation\n(After Fix)', fontsize=12, fontweight='bold')

plt.suptitle('Perpendicular Vector Direction Change\n(Diagonal Beam Example)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/perpendicular_vector_comparison_diagonal.png', dpi=150, bbox_inches='tight')
print("Saved diagram to: /tmp/perpendicular_vector_comparison_diagonal.png")

print("\nVisualization complete!")
print("The change from [-dy, dx] to [dy, -dx] rotates the perpendicular vector")
print("from 90° counter-clockwise to 90° clockwise relative to the beam direction.")
