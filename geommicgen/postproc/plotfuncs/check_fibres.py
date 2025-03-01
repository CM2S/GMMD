import math

def is_fiber_completely_outside_rve(fiber, length=1.0, height=1.0):
    """
    Check if a fiber is completely outside the RVE boundaries.
    Assumes the RVE starts at the origin (0,0) and extends to (length, height).
    
    Parameters:
    - fiber: Dictionary containing center_coordinates and radius
    - length: Length of the RVE in x-direction (default: 1.0)
    - height: Height of the RVE in y-direction (default: 1.0)
    
    Returns:
    - bool: True if fiber is completely outside RVE, False otherwise
    """
    # Extract fiber properties
    center_x = float(fiber['center_coordinates']['x'])
    center_y = float(fiber['center_coordinates']['y'])
    radius = float(fiber['radius'])
    
    # Calculate the closest point of the RVE to the fiber center
    closest_x = max(0, min(center_x, length))
    closest_y = max(0, min(center_y, height))
    
    # Calculate distance from fiber center to closest point on RVE
    distance = math.sqrt((center_x - closest_x)**2 + (center_y - closest_y)**2)
    
    # If distance is greater than radius, fiber is completely outside
    return distance > radius