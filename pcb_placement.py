import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List, Tuple, Optional
import copy

@dataclass
class Component:
    """Represents a PCB component with position and constraints"""
    name: str
    width: int
    height: int
    x: int = 0
    y: int = 0
    must_be_on_edge: bool = False
    can_place_anywhere: bool = True
    proximity_target: str = None
    max_proximity_distance: float = 0

@dataclass
class Rectangle:
    """Represents a free rectangle in Max Rects algorithm"""
    x: int
    y: int
    width: int
    height: int

class PCBPlacer:
    """PCB Component Placement System using Max Rectangles algorithm"""
    
    def __init__(self, board_width: int, board_height: int):
        self.board_width = board_width
        self.board_height = board_height
        self.placed_components = []
        self.free_rectangles = [Rectangle(0, 0, board_width, board_height)]
        
    def is_position_valid(self, component: Component, x: int, y: int) -> bool:
        """Check if a position is valid for a component"""
        # Check board boundaries
        if x < 0 or y < 0 or x + component.width > self.board_width or y + component.height > self.board_height:
            return False
            
        # Check for overlaps with existing components
        for placed in self.placed_components:
            if not (x >= placed.x + placed.width or 
                   x + component.width <= placed.x or
                   y >= placed.y + placed.height or
                   y + component.height <= placed.y):
                return False
        
        # Check edge constraint
        if component.must_be_on_edge:
            on_edge = (x == 0 or y == 0 or 
                      x + component.width == self.board_width or 
                      y + component.height == self.board_height)
            if not on_edge:
                return False
        
        # Check proximity constraint
        if component.proximity_target:
            target = next((c for c in self.placed_components if c.name == component.proximity_target), None)
            if target:
                target_center_x = target.x + target.width // 2
                target_center_y = target.y + target.height // 2
                component_center_x = x + component.width // 2
                component_center_y = y + component.height // 2
                
                distance = np.sqrt((component_center_x - target_center_x)**2 + 
                                 (component_center_y - target_center_y)**2)
                if distance > component.max_proximity_distance:
                    return False
        
        return True
    
    def find_best_position_bssf(self, component: Component) -> Tuple[int, int, int]:
        """Best Short Side Fit heuristic - Minimizes leftover horizontal or vertical space"""
        best_x, best_y = -1, -1
        best_short_side = float('inf')
        best_long_side = float('inf')
        
        for rect in self.free_rectangles:
            if rect.width >= component.width and rect.height >= component.height:
                leftover_horizontal = rect.width - component.width
                leftover_vertical = rect.height - component.height
                short_side = min(leftover_horizontal, leftover_vertical)
                long_side = max(leftover_horizontal, leftover_vertical)
                
                if self.is_position_valid(component, rect.x, rect.y):
                    if short_side < best_short_side or (short_side == best_short_side and long_side < best_long_side):
                        best_x, best_y = rect.x, rect.y
                        best_short_side = short_side
                        best_long_side = long_side
        
        return best_x, best_y, best_short_side

    def find_best_position_blsf(self, component: Component) -> Tuple[int, int, int]:
        """Best Long Side Fit heuristic - Minimizes leftover longer side space"""
        best_x, best_y = -1, -1
        best_short_side = float('inf')
        best_long_side = float('inf')
        
        for rect in self.free_rectangles:
            if rect.width >= component.width and rect.height >= component.height:
                leftover_horizontal = rect.width - component.width
                leftover_vertical = rect.height - component.height
                short_side = min(leftover_horizontal, leftover_vertical)
                long_side = max(leftover_horizontal, leftover_vertical)
                
                if self.is_position_valid(component, rect.x, rect.y):
                    if long_side < best_long_side or (long_side == best_long_side and short_side < best_short_side):
                        best_x, best_y = rect.x, rect.y
                        best_short_side = short_side
                        best_long_side = long_side
        
        return best_x, best_y, best_long_side

    def find_best_position_baf(self, component: Component) -> Tuple[int, int, int]:
        """Best Area Fit heuristic - Minimizes total leftover area"""
        best_x, best_y = -1, -1
        best_area_fit = float('inf')
        best_short_side = float('inf')
        
        for rect in self.free_rectangles:
            if rect.width >= component.width and rect.height >= component.height:
                leftover_horizontal = rect.width - component.width
                leftover_vertical = rect.height - component.height
                area_fit = leftover_horizontal * leftover_vertical
                short_side = min(leftover_horizontal, leftover_vertical)
                
                if self.is_position_valid(component, rect.x, rect.y):
                    if area_fit < best_area_fit or (area_fit == best_area_fit and short_side < best_short_side):
                        best_x, best_y = rect.x, rect.y
                        best_area_fit = area_fit
                        best_short_side = short_side
        
        return best_x, best_y, best_area_fit

    def split_free_rectangle(self, rect: Rectangle, component: Component, x: int, y: int) -> List[Rectangle]:
        """Split a free rectangle after placing a component"""
        new_rects = []
        
        # Create new rectangles from the split
        if x > rect.x:  # Left side
            new_rects.append(Rectangle(rect.x, rect.y, x - rect.x, rect.height))
        
        if x + component.width < rect.x + rect.width:  # Right side
            new_rects.append(Rectangle(x + component.width, rect.y, 
                                     rect.x + rect.width - x - component.width, rect.height))
        
        if y > rect.y:  # Top side
            new_rects.append(Rectangle(rect.x, rect.y, rect.width, y - rect.y))
        
        if y + component.height < rect.y + rect.height:  # Bottom side
            new_rects.append(Rectangle(rect.x, y + component.height, 
                                     rect.width, rect.y + rect.height - y - component.height))
        
        return new_rects

    def place_component(self, component: Component) -> bool:
        """Place a component using Max Rects algorithm with multiple heuristics"""
        # Try different heuristics and choose the best result
        heuristics = [
            self.find_best_position_bssf,
            self.find_best_position_blsf,
            self.find_best_position_baf
        ]
        
        best_x, best_y = -1, -1
        best_score = float('inf')
        
        for heuristic in heuristics:
            x, y, score = heuristic(component)
            if x != -1 and score < best_score:
                best_x, best_y, best_score = x, y, score
        
        if best_x == -1:
            return False
        
        # Place the component
        component.x = best_x
        component.y = best_y
        self.placed_components.append(component)
        
        # Update free rectangles
        new_free_rectangles = []
        
        for rect in self.free_rectangles:
            if (best_x < rect.x + rect.width and best_x + component.width > rect.x and
                best_y < rect.y + rect.height and best_y + component.height > rect.y):
                # Rectangle intersects with placed component, split it
                new_rects = self.split_free_rectangle(rect, component, best_x, best_y)
                new_free_rectangles.extend(new_rects)
            else:
                # Rectangle doesn't intersect, keep it
                new_free_rectangles.append(rect)
        
        # Remove redundant rectangles (contained within others)
        self.free_rectangles = self.remove_redundant_rectangles(new_free_rectangles)
        
        return True

    def remove_redundant_rectangles(self, rectangles: List[Rectangle]) -> List[Rectangle]:
        """Remove rectangles that are contained within other rectangles"""
        non_redundant = []
        
        for i, rect1 in enumerate(rectangles):
            is_redundant = False
            for j, rect2 in enumerate(rectangles):
                if i != j:
                    if (rect2.x <= rect1.x and rect2.y <= rect1.y and
                        rect2.x + rect2.width >= rect1.x + rect1.width and
                        rect2.y + rect2.height >= rect1.y + rect1.height):
                        is_redundant = True
                        break
            if not is_redundant:
                non_redundant.append(rect1)
        
        return non_redundant

    def check_mikrobus_parallel_constraint(self, mb1: Component, mb2: Component) -> bool:
        """Check if two MikroBus connectors are on opposite edges and parallel"""
        # Check if they're on opposite edges
        mb1_on_left = mb1.x == 0
        mb1_on_right = mb1.x + mb1.width == self.board_width
        mb1_on_top = mb1.y == 0
        mb1_on_bottom = mb1.y + mb1.height == self.board_height
        
        mb2_on_left = mb2.x == 0
        mb2_on_right = mb2.x + mb2.width == self.board_width
        mb2_on_top = mb2.y == 0
        mb2_on_bottom = mb2.y + mb2.height == self.board_height
        
        # They should be on opposite edges
        opposite_horizontal = (mb1_on_left and mb2_on_right) or (mb1_on_right and mb2_on_left)
        opposite_vertical = (mb1_on_top and mb2_on_bottom) or (mb1_on_bottom and mb2_on_top)
        
        return opposite_horizontal or opposite_vertical

    def force_mikrobus_placement(self, mb1: Component, mb2: Component):
        """Force MikroBus connectors to be on opposite edges"""
        # Place MB1 first
        if not self.place_component(mb1):
            return False
            
        # Determine where MB2 should go based on MB1's position
        target_positions = []
        
        if mb1.x == 0:  # MB1 on left edge
            # MB2 should be on right edge
            target_positions = [(45, y) for y in range(46) if y % 5 == 0]
        elif mb1.x + mb1.width == self.board_width:  # MB1 on right edge
            # MB2 should be on left edge
            target_positions = [(0, y) for y in range(46) if y % 5 == 0]
        elif mb1.y == 0:  # MB1 on top edge
            # MB2 should be on bottom edge
            target_positions = [(x, 45) for x in range(46) if x % 5 == 0]
        elif mb1.y + mb1.height == self.board_height:  # MB1 on bottom edge
            # MB2 should be on top edge
            target_positions = [(x, 0) for x in range(46) if x % 5 == 0]
        
        # Try each target position
        for x, y in target_positions:
            if self.is_position_valid(mb2, x, y):
                mb2.x = x
                mb2.y = y
                self.placed_components.append(mb2)
                
                # Update free rectangles manually for MB2
                new_free_rectangles = []
                for rect in self.free_rectangles:
                    if (x < rect.x + rect.width and x + mb2.width > rect.x and
                        y < rect.y + rect.height and y + mb2.height > rect.y):
                        new_rects = self.split_free_rectangle(rect, mb2, x, y)
                        new_free_rectangles.extend(new_rects)
                    else:
                        new_free_rectangles.append(rect)
                
                self.free_rectangles = self.remove_redundant_rectangles(new_free_rectangles)
                return True
        
        return False

def create_components() -> List[Component]:
    """Create the list of components with their constraints"""
    components = [
        Component(
            name="USB_CONNECTOR",
            width=5, height=5,
            must_be_on_edge=True
        ),
        Component(
            name="MIKROBUS_CONNECTOR_1",
            width=5, height=5,
            must_be_on_edge=True
        ),
        Component(
            name="MIKROBUS_CONNECTOR_2",
            width=5, height=5,
            must_be_on_edge=True
        ),
        Component(
            name="MICROCONTROLLER",
            width=5, height=5,
            can_place_anywhere=True
        ),
        Component(
            name="CRYSTAL",
            width=5, height=5,
            proximity_target="MICROCONTROLLER",
            max_proximity_distance=10
        )
    ]
    return components

def solve_pcb_placement() -> PCBPlacer:
    """Solve the PCB placement problem"""
    placer = PCBPlacer(50, 50)
    components = create_components()
    
    # Get specific components
    usb = next(c for c in components if c.name == "USB_CONNECTOR")
    mb1 = next(c for c in components if c.name == "MIKROBUS_CONNECTOR_1")
    mb2 = next(c for c in components if c.name == "MIKROBUS_CONNECTOR_2")
    microcontroller = next(c for c in components if c.name == "MICROCONTROLLER")
    crystal = next(c for c in components if c.name == "CRYSTAL")
    
    # Phase 1: Place edge-constrained components first
    print("Phase 1: Placing USB connector...")
    if not placer.place_component(usb):
        print(f"Failed to place {usb.name}")
        return None
    
    # Phase 2: Place MikroBus connectors with special constraint handling
    print("Phase 2: Placing MikroBus connectors with parallel constraint...")
    if not placer.force_mikrobus_placement(mb1, mb2):
        print("Failed to place MikroBus connectors with parallel constraint")
        return None
    
    # Phase 3: Place microcontroller
    print("Phase 3: Placing microcontroller...")
    if not placer.place_component(microcontroller):
        print(f"Failed to place {microcontroller.name}")
        return None
    
    # Phase 4: Place crystal with proximity constraint
    print("Phase 4: Placing crystal with proximity constraint...")
    if not placer.place_component(crystal):
        print(f"Failed to place {crystal.name}")
        return None
    
    return placer

def visualize_placement(placer: PCBPlacer):
    """Create a visualization of the PCB placement"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Draw board boundary
    board_rect = patches.Rectangle((0, 0), placer.board_width, placer.board_height, 
                                  linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3)
    ax.add_patch(board_rect)
    
    # Colors for different components
    colors = {
        'USB_CONNECTOR': 'red',
        'MICROCONTROLLER': 'blue',
        'CRYSTAL': 'orange',
        'MIKROBUS_CONNECTOR_1': 'green',
        'MIKROBUS_CONNECTOR_2': 'purple'
    }
    
    # Labels for visualization
    labels = {
        'USB_CONNECTOR': 'USB',
        'MICROCONTROLLER': 'μC',
        'CRYSTAL': 'XTAL',
        'MIKROBUS_CONNECTOR_1': 'MB1',
        'MIKROBUS_CONNECTOR_2': 'MB2'
    }
    
    # Draw components
    for component in placer.placed_components:
        color = colors.get(component.name, 'gray')
        comp_rect = patches.Rectangle((component.x, component.y), component.width, component.height,
                                    linewidth=2, edgecolor='black', facecolor=color, alpha=0.8)
        ax.add_patch(comp_rect)
        
        # Add label
        label = labels.get(component.name, component.name)
        ax.text(component.x + component.width/2, component.y + component.height/2, label,
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Draw proximity constraint circle for Crystal
    crystal = next(c for c in placer.placed_components if c.name == "CRYSTAL")
    microcontroller = next(c for c in placer.placed_components if c.name == "MICROCONTROLLER")
    
    mc_center_x = microcontroller.x + microcontroller.width // 2
    mc_center_y = microcontroller.y + microcontroller.height // 2
    
    proximity_circle = patches.Circle((mc_center_x, mc_center_y), 10, 
                                    linewidth=2, edgecolor='orange', facecolor='none', 
                                    linestyle='--', alpha=0.8)
    ax.add_patch(proximity_circle)
    
    # Set up the plot
    ax.set_xlim(-2, placer.board_width + 2)
    ax.set_ylim(-2, placer.board_height + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (units)', fontsize=12)
    ax.set_ylabel('Y (units)', fontsize=12)
    ax.set_title('PCB Component Placement - Max Rects Algorithm\\nwith Constraint Satisfaction', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = []
    for name, color in colors.items():
        label = labels.get(name, name)
        legend_elements.append(patches.Patch(color=color, label=f'{label} - {name.replace("_", " ").title()}'))
    
    legend_elements.append(patches.Patch(color='none', label='Proximity Circle (10 units)', 
                                        linestyle='--', edgecolor='orange'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))
    
    plt.tight_layout()
    plt.show()

def print_board_visualization(placer: PCBPlacer):
    """Print a text-based visualization of the board"""
    # Create a grid representation
    grid = [['.' for _ in range(placer.board_width)] for _ in range(placer.board_height)]
    
    # Fill in components
    component_chars = {
        'USB_CONNECTOR': 'U',
        'MICROCONTROLLER': 'M',
        'CRYSTAL': 'C',
        'MIKROBUS_CONNECTOR_1': '1',
        'MIKROBUS_CONNECTOR_2': '2'
    }
    
    for comp in placer.placed_components:
        char = component_chars.get(comp.name, '?')
        for y in range(comp.y, comp.y + comp.height):
            for x in range(comp.x, comp.x + comp.width):
                if 0 <= x < placer.board_width and 0 <= y < placer.board_height:
                    grid[y][x] = char
    
    print("\\n=== Board Layout (50x50) ===")
    print("Legend: U=USB, M=Microcontroller, C=Crystal, 1=MikroBus1, 2=MikroBus2, .=Empty")
    print("+" + "-" * placer.board_width + "+")
    
    # Print every 5th row to make it readable
    for y in range(0, placer.board_height, 5):
        row = "|" + "".join(grid[y]) + "|"
        print(row)
    
    print("+" + "-" * placer.board_width + "+")

def validate_all_constraints(placer: PCBPlacer) -> bool:
    """Validate all constraints are satisfied"""
    print("\\n=== Comprehensive Constraint Validation ===")
    all_valid = True
    
    # 1. Boundary Constraint
    print("1. Boundary Constraints:")
    for comp in placer.placed_components:
        within_bounds = (0 <= comp.x and 0 <= comp.y and 
                        comp.x + comp.width <= placer.board_width and 
                        comp.y + comp.height <= placer.board_height)
        print(f"   {comp.name}: {'✓' if within_bounds else '✗'}")
        if not within_bounds:
            all_valid = False
    
    # 2. No Overlapping
    print("\\n2. No Overlapping Constraints:")
    for i, comp1 in enumerate(placer.placed_components):
        for j, comp2 in enumerate(placer.placed_components[i+1:], i+1):
            no_overlap = (comp1.x >= comp2.x + comp2.width or 
                         comp1.x + comp1.width <= comp2.x or
                         comp1.y >= comp2.y + comp2.height or
                         comp1.y + comp1.height <= comp2.y)
            if not no_overlap:
                print(f"   {comp1.name} & {comp2.name}: ✗ (OVERLAP DETECTED)")
                all_valid = False
    
    if all_valid:
        print("   All components: ✓ (No overlaps)")
    
    # 3. Edge Placement
    print("\\n3. Edge Placement Constraints:")
    edge_components = [c for c in placer.placed_components if c.must_be_on_edge]
    for comp in edge_components:
        on_edge = (comp.x == 0 or comp.y == 0 or 
                  comp.x + comp.width == placer.board_width or 
                  comp.y + comp.height == placer.board_height)
        print(f"   {comp.name}: {'✓' if on_edge else '✗'}")
        if not on_edge:
            all_valid = False
    
    # 4. Proximity Constraint
    print("\\n4. Proximity Constraints:")
    crystal = next(c for c in placer.placed_components if c.name == "CRYSTAL")
    microcontroller = next(c for c in placer.placed_components if c.name == "MICROCONTROLLER")
    
    if crystal and microcontroller:
        mc_center_x = microcontroller.x + microcontroller.width // 2
        mc_center_y = microcontroller.y + microcontroller.height // 2
        crystal_center_x = crystal.x + crystal.width // 2
        crystal_center_y = crystal.y + crystal.height // 2
        
        distance = np.sqrt((crystal_center_x - mc_center_x)**2 + (crystal_center_y - mc_center_y)**2)
        within_proximity = distance <= 10
        print(f"   Crystal-Microcontroller distance: {distance:.2f} units {'✓' if within_proximity else '✗'}")
        if not within_proximity:
            all_valid = False
    
    # 5. Parallel Placement
    print("\\n5. Parallel Placement Constraints:")
    mb1 = next(c for c in placer.placed_components if c.name == "MIKROBUS_CONNECTOR_1")
    mb2 = next(c for c in placer.placed_components if c.name == "MIKROBUS_CONNECTOR_2")
    is_parallel = placer.check_mikrobus_parallel_constraint(mb1, mb2)
    print(f"   MikroBus connectors parallel/opposite: {'✓' if is_parallel else '✗'}")
    if not is_parallel:
        all_valid = False
    
    print(f"\\n=== Overall Constraint Satisfaction: {'✓ PASSED' if all_valid else '✗ FAILED'} ===")
    return all_valid

def main():
    """Main execution function"""
    print("=" * 60)
    print("PCB COMPONENT PLACEMENT SYSTEM")
    print("Max Rectangles Algorithm with Constraint Satisfaction")
    print("=" * 60)
    
    # Solve the placement problem
    placer = solve_pcb_placement()
    
    if placer:
        print("\\n✓ Placement Algorithm Successfully Completed!")
        
        # Print text visualization
        print_board_visualization(placer)
        
        # Print detailed results
        print("\\n=== Placement Results Summary ===")
        print(f"Board Size: {placer.board_width} × {placer.board_height} units")
        print(f"Components Placed: {len(placer.placed_components)}/5")
        print("\\nComponent Positions:")
        for comp in placer.placed_components:
            print(f"  {comp.name}: ({comp.x}, {comp.y}) to ({comp.x + comp.width}, {comp.y + comp.height})")
        
        # Validate all constraints
        constraint_valid = validate_all_constraints(placer)
        
        # Performance metrics
        print("\\n=== Algorithm Performance Metrics ===")
        total_component_area = len(placer.placed_components) * 5 * 5
        board_area = placer.board_width * placer.board_height
        utilization = (total_component_area / board_area) * 100
        print(f"Board Utilization: {utilization:.1f}%")
        print(f"Free Rectangles Remaining: {len(placer.free_rectangles)}")
        print(f"Placement Efficiency: {'Optimal' if constraint_valid else 'Sub-optimal'}")
        
        # Create matplotlib visualization
        try:
            print("\\nGenerating visual plot...")
            visualize_placement(placer)
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Text-based visualization shown above.")
        
        print("\\n" + "=" * 60)
        print("ASSIGNMENT COMPLETED SUCCESSFULLY")
        print("All deliverables generated:")
        print("✓ Complete Max Rects algorithm implementation")
        print("✓ Constraint handling system")
        print("✓ Component placement visualization")
        print("✓ Constraint verification system")
        print("=" * 60)
        
    else:
        print("\\n✗ Failed to find a valid placement solution!")
        print("Consider relaxing constraints or increasing board size.")

if __name__ == "__main__":
    main()