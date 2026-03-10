import threading
import math
try:
    import pygame
    from pygame.locals import K_ESCAPE, K_TAB, K_c
except ImportError:
    pass

import numpy as np
import carla

class CameraDisplay:
    """Receives CARLA camera images and converts them for Pygame."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = None
        self._lock = threading.Lock()

    def callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        with self._lock:
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display):
        with self._lock:
            if self.surface is not None:
                display.blit(self.surface, (0, 0))

class PygameVisualizer:
    """Manages the Pygame display and HUD overlay for the ADAS Supervisor."""
    
    COLORS = {
        'SAFE':    (46, 204, 113),
        'CAUTION': (241, 196, 15),
        'DANGER':  (231, 76, 60),
    }

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("ADAS Supervision Framework")
        
        self._font_big = pygame.font.SysFont('consolas', 22, bold=True)
        self._font = pygame.font.SysFont('consolas', 16)
        self._font_small = pygame.font.SysFont('consolas', 13)
        self._font_alert = pygame.font.SysFont('consolas', 28, bold=True)
        
        self.camera_display = CameraDisplay(width, height)
        self._show_hud = True
        
        # Telemetry State
        self.risk = 0.0
        self.takeover_stage = 0
        self.speed = 0.0
        self.ttc = 0.0
        self.confidence = 0.0
        self.detections = 0
        self.run_state = "RUNNING"
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        

        
    def attach_camera(self, world, vehicle):
        """Attaches spectator RGB cameras to the ego vehicle."""
        self._world = world
        self._vehicle = vehicle
        
        bp_library = world.get_blueprint_library()
        self._camera_bp = bp_library.find('sensor.camera.rgb')
        self._camera_bp.set_attribute('image_size_x', str(self.width))
        self._camera_bp.set_attribute('image_size_y', str(self.height))
        self._camera_bp.set_attribute('fov', '90')
        
        # Define multiple viewing angles (Name, Transform)
        self.camera_configs = [
            ("Third Person", carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))),
            ("Driver Seat", carla.Transform(carla.Location(x=0.2, y=-0.3, z=1.3), carla.Rotation(pitch=0))),
            ("Passenger Seat", carla.Transform(carla.Location(x=0.2, y=0.35, z=1.3), carla.Rotation(pitch=0))),
            ("Bird's Eye", carla.Transform(carla.Location(x=0.0, z=15.0), carla.Rotation(pitch=-90))),
        ]
        
        self.current_camera_idx = 0
        name, transform = self.camera_configs[self.current_camera_idx]
        
        try:
            self.camera_sensor = self._world.spawn_actor(self._camera_bp, transform, attach_to=self._vehicle)
            self.camera_sensor.listen(self.camera_display.callback)
        except Exception as e:
            print(f"Warning: Failed to spawn primary camera: {e}")
            self.camera_sensor = None

    def _switch_camera(self):
        """Immediately changes the viewing angle of the active camera."""
        if not getattr(self, 'camera_sensor', None) or not self.camera_sensor.is_alive:
            return
            
        self.current_camera_idx = (self.current_camera_idx + 1) % len(self.camera_configs)
        name, transform = self.camera_configs[self.current_camera_idx]
        
        # Instantly move the existing camera relative to the ego vehicle
        self.camera_sensor.set_transform(transform)

    def update_telemetry(self, risk, takeover_stage, speed, ttc, confidence, detections, run_state, control):
        self.risk = risk
        self.takeover_stage = takeover_stage
        self.speed = speed
        self.ttc = ttc
        self.confidence = confidence
        self.detections = detections
        self.run_state = run_state
        if control:
            self.steer = control.steer
            self.throttle = control.throttle
            self.brake = control.brake
            
    def tick(self):
        """Processes pygame events and updates the display."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_TAB:
                    self._show_hud = not self._show_hud
                elif event.key == K_c:
                    # Cycle through camera views
                    self._switch_camera()

                    
        # Draw camera background
        self.camera_display.render(self.display)
        
        # Draw HUD overlays
        if self._show_hud:
            self._render_hud()
            
        pygame.display.flip()
        return True

    def _render_hud(self):
        # 1. State bar
        bar_h = 40
        bar_surface = pygame.Surface((self.width, bar_h))
        bar_surface.set_alpha(180)
        
        # Determine color based on stage
        stage_colors = {
            0: self.COLORS['SAFE'],
            1: self.COLORS['CAUTION'],
            2: (230, 126, 34),     # Speed reduction (orange)
            3: self.COLORS['DANGER'], # Takeover
            4: (150, 0, 0)         # Emergency brake
        }
        color = stage_colors.get(self.takeover_stage, self.COLORS['SAFE'])
        bar_surface.fill(color)
        self.display.blit(bar_surface, (0, 0))
        
        stage_names = ["0: NORMAL", "1: WARNING", "2: SPEED REDUCTION", "3: TAKEOVER REQ", "4: EMERGENCY BRAKE"]
        stage_text = stage_names[self.takeover_stage] if 0 <= self.takeover_stage <= 4 else "UNKNOWN"
        camera_name = self.camera_configs[self.current_camera_idx][0] if hasattr(self, 'camera_configs') else ""
        
        status_text = f"  STATE: {self.run_state}   |   STAGE: {stage_text}   |   VIEW: {camera_name}"
        text_surf = self._font_big.render(status_text, True, (255, 255, 255))
        self.display.blit(text_surf, (8, 8))

        # 2. Telemetry Panel
        panel_lines = [
            f"Speed:       {self.speed * 3.6:6.1f} km/h",
            f"Throttle:    {self.throttle:6.2f}",
            f"Brake:       {self.brake:6.2f}",
            f"Steer:       {self.steer:+6.3f}",
            f"",
            f"Risk:        {self.risk:6.3f}",
            f"Min TTC:     {self.ttc:6.2f} s" if self.ttc != float("inf") else "Min TTC:        -- s",
            f"Objects:     {self.detections:4d}",
            f"Confidence:  {self.confidence:6.2f}",
        ]
        
        panel_w = 280
        panel_h = len(panel_lines) * 20 + 16
        panel_surface = pygame.Surface((panel_w, panel_h))
        panel_surface.set_alpha(150)
        panel_surface.fill((20, 20, 20))
        self.display.blit(panel_surface, (0, bar_h))

        y = bar_h + 8
        for line in panel_lines:
            line_surf = self._font.render(line, True, (220, 220, 220))
            self.display.blit(line_surf, (10, y))
            y += 20
            
        # 3. Giant alert text for Takeover Request and Emergency Brake
        if self.takeover_stage >= 3:
            alert_text = ">> EMERGENCY BRAKING <<" if self.takeover_stage == 4 else ">> TAKE CONTROL NOW <<"
            alert_surf = self._font_alert.render(alert_text, True, (255, 255, 255))
            alert_bg = pygame.Surface((alert_surf.get_width() + 40, alert_surf.get_height() + 20))
            alert_bg.set_alpha(200)
            alert_bg.fill((231, 76, 60))
            x = (self.width - alert_bg.get_width()) // 2
            self.display.blit(alert_bg, (x, y))
            self.display.blit(alert_surf, (x + 20, y + 10))
            


    def cleanup(self):
        if hasattr(self, 'camera_sensor') and self.camera_sensor is not None:
            if self.camera_sensor.is_alive:
                self.camera_sensor.stop()
                self.camera_sensor.destroy()
            self.camera_sensor = None
        
        # Pump event queue to clear SDL locks
        try:
            pygame.event.pump()
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
