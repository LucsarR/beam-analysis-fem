"""
Create a visual mockup showing the reactions display in the app.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, '⚙️ Reaction Forces at Constraints', 
        ha='center', va='top', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

# Table header
header_y = 8.5
ax.add_patch(FancyBboxPatch((0.5, header_y-0.4), 9, 0.5, 
                            boxstyle="round,pad=0.05", 
                            facecolor='#4CAF50', edgecolor='black', linewidth=1.5))

headers = ['Node', 'X', 'Y', 'Direction', 'Reaction', 'Unit']
header_positions = [1, 2.3, 3.6, 5, 7, 8.5]
for i, (header, pos) in enumerate(zip(headers, header_positions)):
    ax.text(pos, header_y-0.15, header, ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')

# Sample data rows
data = [
    ['1', '0.0000', '0.0000', 'X', '0.000000e+00', 'N'],
    ['1', '0.0000', '0.0000', 'Y', '1.000000e+03', 'N'],
    ['1', '0.0000', '0.0000', 'Rotation', '1.000000e+03', 'N·m'],
]

row_y = 7.8
for i, row_data in enumerate(data):
    row_y_pos = row_y - i * 0.6
    
    # Alternating row colors
    if i % 2 == 0:
        ax.add_patch(FancyBboxPatch((0.5, row_y_pos-0.4), 9, 0.5, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#E8F5E9', edgecolor='gray', linewidth=0.5))
    else:
        ax.add_patch(FancyBboxPatch((0.5, row_y_pos-0.4), 9, 0.5, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='white', edgecolor='gray', linewidth=0.5))
    
    for j, (value, pos) in enumerate(zip(row_data, header_positions)):
        color = 'black'
        weight = 'normal'
        if j == 4:  # Highlight reaction values
            if 'e+03' in value:
                color = '#1976D2'
                weight = 'bold'
        ax.text(pos, row_y_pos-0.15, value, ha='center', va='center', 
                fontsize=10, color=color, weight=weight)

# Info box
info_y = 5.5
ax.add_patch(FancyBboxPatch((0.5, info_y-0.6), 9, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=1.5))
ax.text(5, info_y-0.2, 'ℹ️ Positive reactions indicate forces in positive coordinate directions.', 
        ha='center', va='center', fontsize=10, style='italic')

# Download button
button_y = 4.3
ax.add_patch(FancyBboxPatch((3, button_y-0.4), 4, 0.6, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#2196F3', edgecolor='black', linewidth=1.5))
ax.text(5, button_y-0.1, '📥 Download Reactions CSV', 
        ha='center', va='center', fontsize=11, color='white', fontweight='bold')

# Section divider
ax.plot([0.5, 9.5], [3.5, 3.5], 'k-', linewidth=2, alpha=0.3)

# Previous section indicator (Displacements above)
ax.text(5, 3, '↑ Nodal Displacements section above ↑', 
        ha='center', va='center', fontsize=9, style='italic', color='gray')

# Next section indicator
ax.text(5, 2.5, '↓ Results tab link below ↓', 
        ha='center', va='center', fontsize=9, style='italic', color='gray')

# Feature highlights
features_y = 1.5
features = [
    '✓ Shows all constrained DOFs',
    '✓ Displays node position',
    '✓ Indicates direction (X/Y/Rotation)',
    '✓ Shows force magnitude with units',
    '✓ Exportable to CSV'
]

for i, feature in enumerate(features):
    ax.text(1, features_y - i*0.3, feature, ha='left', va='center', 
            fontsize=9, color='#1B5E20')

plt.title('Reactions Display in Streamlit Application', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/tmp/reactions_display_mockup.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Created mockup: /tmp/reactions_display_mockup.png")

# Create second figure showing the aspect ratio fix
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Before fix - distorted axes
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 3)  # Different scale
ax1.set_aspect('auto')

# Draw 45-degree beam
ax1.plot([0.2, 1.2], [0.5, 1.5], 'k-', linewidth=4, label='Beam (45°)')
ax1.scatter([0.2, 1.2], [0.5, 1.5], c='blue', s=100, zorder=5)

# Draw perpendicular vector (appears wrong due to scaling)
perp_x, perp_y = -1.0, 1.0  # Perpendicular to 45° line
norm = (perp_x**2 + perp_y**2)**0.5
perp_x, perp_y = perp_x/norm, perp_y/norm
mid_x, mid_y = 0.7, 1.0

ax1.arrow(mid_x, mid_y, perp_x*0.4, perp_y*0.4, 
          head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
ax1.text(mid_x + perp_x*0.5, mid_y + perp_y*0.5, 'Fill\ndirection', 
         ha='center', fontsize=10, color='red', fontweight='bold')

ax1.text(1.5, 0.3, '❌ Visual angle\nlooks wrong', ha='center', fontsize=11, 
         color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax1.set_title('Before Fix: Auto Scaling\n(Axes have different scales)', 
              fontsize=12, fontweight='bold', color='red')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')

# After fix - equal aspect ratio
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)  # Same scale
ax2.set_aspect('equal')

# Draw 45-degree beam
ax2.plot([0.2, 1.2], [0.2, 1.2], 'k-', linewidth=4, label='Beam (45°)')
ax2.scatter([0.2, 1.2], [0.2, 1.2], c='blue', s=100, zorder=5)

# Draw perpendicular vector (now appears correct)
perp_x, perp_y = -1.0, 1.0
norm = (perp_x**2 + perp_y**2)**0.5
perp_x, perp_y = perp_x/norm, perp_y/norm
mid_x, mid_y = 0.7, 0.7

ax2.arrow(mid_x, mid_y, perp_x*0.4, perp_y*0.4, 
          head_width=0.1, head_length=0.07, fc='green', ec='green', linewidth=2)
ax2.text(mid_x + perp_x*0.5, mid_y + perp_y*0.5, 'Fill\ndirection\n(90°)', 
         ha='center', fontsize=10, color='green', fontweight='bold')

ax2.text(1.5, 0.3, '✓ True 90°\nangle', ha='center', fontsize=11, 
         color='green', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax2.set_title('After Fix: Equal Aspect Ratio\n(scaleanchor="x", scaleratio=1)', 
              fontsize=12, fontweight='bold', color='green')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')

plt.suptitle('Equal Aspect Ratio Fix for Inclined Beams', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/aspect_ratio_fix_comparison.png', dpi=150, bbox_inches='tight')
print("Created comparison: /tmp/aspect_ratio_fix_comparison.png")

print("\n✓ Mockups created successfully!")
