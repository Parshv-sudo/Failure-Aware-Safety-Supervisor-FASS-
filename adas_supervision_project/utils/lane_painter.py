"""
Utility to procedurally paint lane markings on OpenDRIVE maps 
in CARLA using the DebugHelper. Useful for OSM-imported maps 
which lack texture rendering.
"""

import carla
import logging

logger = logging.getLogger(__name__)

class LanePainter:
    """Uses CARLA's DebugHelper to draw persistent lines along lane boundaries."""

    def __init__(self, world: "carla.World", config: dict = None):
        self.world = world
        self.map = world.get_map()
        self.debug = world.debug
        self.cfg = config or {}
        
        self.precision = self.cfg.get("line_precision", 2.0)
        self.thickness = self.cfg.get("line_thickness", 0.1)
        self.z_offset = self.cfg.get("z_offset", 0.05)
        
        self.color_white = carla.Color(r=255, g=255, b=255)
        self.color_yellow = carla.Color(r=255, g=200, b=0)

    def paint_lines(self):
        """Iterates over the topology and paints lines for every lane."""
        logger.info("Initializing LanePainter to draw road lines...")
        
        # Get topology (list of tuples of waypoints: (start, end))
        topology = self.map.get_topology()
        
        drawn_count = 0
        
        for wp_pair in topology:
            # We follow the lane from start to end by generating waypoints every self.precision meters
            wp = wp_pair[0]
            wp_end = wp_pair[1]
            
            # Distance of the segment
            distance = wp.transform.location.distance(wp_end.transform.location)
            
            if distance <= 0.0:
                continue
                
            # Iterate through the lane
            current_wp = wp
            while current_wp is not None:
                next_wps = current_wp.next(self.precision)
                if not next_wps:
                    break
                    
                next_wp = next_wps[0] # Pick the driving continuity path
                
                # Check if we passed the designated end waypoint for this topology segment
                if current_wp.transform.location.distance(wp_end.transform.location) < self.precision:
                    next_wp = wp_end
                    
                self._draw_lane_boundaries(current_wp, next_wp)
                drawn_count += 2
                
                if next_wp.id == wp_end.id:
                    break
                    
                current_wp = next_wp

        logger.info(f"LanePainter finished drawing ~{drawn_count} line segments.")

    def _draw_lane_boundaries(self, wp_start, wp_end):
        """Draws the left and right lane boundaries between two waypoints."""
        
        # Left Boundary
        left_start = self._get_boundary_location(wp_start, True)
        left_end = self._get_boundary_location(wp_end, True)
        color = self._get_marking_color(wp_start.left_lane_marking)
        
        # In CARLA, a 'None' marking type implies no physical marking
        if wp_start.left_lane_marking.type != carla.LaneMarkingType.NONE:
            self._draw_line(left_start, left_end, color)
            
        # Right Boundary
        right_start = self._get_boundary_location(wp_start, False)
        right_end = self._get_boundary_location(wp_end, False)
        color = self._get_marking_color(wp_start.right_lane_marking)
        
        if wp_start.right_lane_marking.type != carla.LaneMarkingType.NONE:
            self._draw_line(right_start, right_end, color)

    def _get_boundary_location(self, waypoint, is_left: bool):
        """Calculates the physical world location of the left or right lane boundary."""
        # Get right vector (points from center to right edge)
        right_vector = waypoint.transform.get_right_vector()
        
        # Half of the lane width
        offset = waypoint.lane_width / 2.0
        
        if is_left:
            offset = -offset
            
        location = waypoint.transform.location + (right_vector * offset)
        location.z += self.z_offset
        return location

    def _get_marking_color(self, marking):
        """Returns the carla.Color based on the lane marking standard."""
        if marking.color == carla.LaneMarkingColor.Yellow:
            return self.color_yellow
        else:
            return self.color_white
            
    def _draw_line(self, start, end, color):
        """Draws a persistent line between two points."""
        self.debug.draw_line(
            start, 
            end, 
            thickness=self.thickness, 
            color=color, 
            life_time=0.0,  # Persistent
            persistent_lines=True
        )
