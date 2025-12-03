import heapq
import numpy as np
import math
from typing import Tuple, List, Optional

class AStarAgent:
    """Agent that plans a path to food using A* with Manhattan heuristic."""

    # Action mappings
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # Direction deltas: [x_delta, y_delta]
    DIRECTIONS = {
        UP: (0, -1),
        RIGHT: (1, 0),
        DOWN: (0, 1),
        LEFT: (-1, 0),
    }

    def __init__(self, action_space):
        self.action_space = action_space
        self.prev_action = self.DOWN # Default starting inertia
        
        # Cache unit size to prevent frame-to-frame fluctuation
        self.cached_unit_size = None

    def select_action(self, obs, info):
        """
        Select action using A* pathfinding to nearest food.
        """
        # 1. Infer Grid Metrics Robustly
        self.current_unit_size = self._get_robust_unit_size(obs)
        self.current_grid_size = self.get_grid_size(obs, self.current_unit_size)
        
        # 2. Parse State
        snake_head = self.get_snake_head(obs, self.current_unit_size)
        food_list = self.get_food_list(obs, self.current_unit_size)
        
        # FIX: Use center-point sampling to ignore visual connectors
        obstacles = self.get_obstacles(obs, self.current_unit_size, self.current_grid_size)
        
        # Explicitly remove head from obstacles (fixes connector overlap issues)
        if snake_head in obstacles:
            obstacles.remove(snake_head)

        # 3. Decision Logic
        action = None
        
        if snake_head is not None and food_list:
            # Find nearest food
            nearest_food = min(food_list, key=lambda f: self.manhattan_distance(snake_head, f))
            
            # Plan path
            path = self.a_star(snake_head, nearest_food, obstacles, self.current_grid_size)
            
            if path and len(path) > 1:
                next_pos = path[1]
                dx = next_pos[0] - snake_head[0]
                dy = next_pos[1] - snake_head[1]
                action = self._get_action_from_delta(dx, dy)

        # 4. Fallback / Safety Check
        # If A* failed or returned a move that is now invalid:
        if action is None or not self._is_valid_move(snake_head, action, self.current_grid_size, obstacles):
            action = self._find_safe_move(snake_head, self.current_grid_size, obstacles)

        # Update inertia
        self.prev_action = action
        return action

    def _get_robust_unit_size(self, obs: np.ndarray) -> int:
        """
        Calculates unit size based on the Snake Head area.
        The Head is the most consistent visual element (255,0,0 square).
        """
        if self.cached_unit_size is not None:
            return self.cached_unit_size

        # Find all head pixels
        head_pixels = np.sum((obs[:, :, 0] == 255) & (obs[:, :, 1] == 0) & (obs[:, :, 2] == 0))
        
        if head_pixels == 0:
            return 10 # Default fallback
            
        # Grid draws squares with a small gap (usually 1px)
        # Area = (size - gap)^2. Approximating size = sqrt(Area) + gap
        # We round to nearest integer.
        estimated_size = math.sqrt(head_pixels) + 1
        
        # Snap to common grid sizes (optional, but helps stability)
        self.cached_unit_size = int(round(estimated_size))
        return self.cached_unit_size

    def get_obstacles(self, obs: np.ndarray, unit_size: int, grid_size: Tuple[int, int]) -> set:
        """
        Detects obstacles by sampling the CENTER of each grid cell.
        This prevents detecting the "connectors" between cells as obstacles in the wrong cell.
        """
        obstacles = set()
        h, w = grid_size
        
        # Pre-calculate half unit to find center
        half_unit = unit_size // 2
        
        for gx in range(w):
            for gy in range(h):
                # Calculate pixel coordinates of the center of this grid cell
                px = gx * unit_size + half_unit
                py = gy * unit_size + half_unit
                
                # Safety bounds check
                if py >= obs.shape[0] or px >= obs.shape[1]:
                    continue

                # Check pixel color
                # Body: R > 0 (can be 1 or 255), G=0, B=0
                r, g, b = obs[py, px]
                
                # Strict check for Body Color (approx [1, 0, 0]) or Head Color
                # We treat anything "Reddish" and not background/food as obstacle
                if r > 0 and g == 0 and b == 0:
                    obstacles.add((gx, gy))
                    
        return obstacles

    def _find_safe_move(self, pos: Tuple[int, int], grid_size: Tuple[int, int], obstacles: set) -> int:
        """
        Greedy fallback. Ensures we don't perform a 180-degree death turn.
        """
        if pos is None:
            return self.DOWN
        
        best_score = -1
        best_action = self.prev_action # Default to inertia
        
        # Define opposite of previous action (to avoid 180 turns)
        opposites = {self.UP: self.DOWN, self.DOWN: self.UP, self.LEFT: self.RIGHT, self.RIGHT: self.LEFT}
        forbidden_action = opposites.get(self.prev_action, -1)

        for action in [self.UP, self.RIGHT, self.DOWN, self.LEFT]:
            # Don't reverse into own neck unless it's the ONLY option
            if action == forbidden_action:
                continue

            if self._is_valid_move(pos, action, grid_size, obstacles):
                next_pos = (pos[0] + self.DIRECTIONS[action][0], pos[1] + self.DIRECTIONS[action][1])
                
                # Lookahead scoring
                routes = self._count_escape_routes(next_pos, grid_size, obstacles)
                score = routes
                
                if score > best_score:
                    best_score = score
                    best_action = action
        
        # If we are trapped, we might have to take the forbidden action or just die
        return best_action

    def get_snake_head(self, obs: np.ndarray, unit_size: int) -> Optional[Tuple[int, int]]:
        # Find red pixels
        coords = np.argwhere(obs[:, :, 0] == 255)
        if coords.size == 0: return None
        
        # Use center of the blob for better accuracy
        y_center = int(np.mean(coords[:, 0]))
        x_center = int(np.mean(coords[:, 1]))
        
        return (x_center // unit_size, y_center // unit_size)

    def get_food_list(self, obs: np.ndarray, unit_size: int) -> List[Tuple[int, int]]:
        # Blue pixels
        coords = np.argwhere(obs[:, :, 2] == 255)
        if coords.size == 0: return []
        
        food_set = set()
        # Sampling optimization: Don't process every pixel, just centers logic roughly
        # Or just use the first pixel of every blob. 
        # Since food is 1x1 grid unit usually, standard integer division works fine 
        # provided we handle the blob correctly.
        
        for y, x in coords:
            # We can just add them all and set() handles duplicates
            # But calculating center is safer. 
            # Simple division is okay for food as it has no connectors.
            food_set.add((x // unit_size, y // unit_size))
            
        return list(food_set)

    def get_grid_size(self, obs: np.ndarray, unit_size: int) -> Tuple[int, int]:
        height, width = obs.shape[:2]
        return (width // unit_size, height // unit_size)

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_action_from_delta(self, dx: int, dy: int) -> int:
        if dx == 1: return self.RIGHT
        elif dx == -1: return self.LEFT
        elif dy == -1: return self.UP
        elif dy == 1: return self.DOWN
        return self.DOWN

    def _is_valid_move(self, pos: Tuple[int, int], action: int, grid_size: Tuple[int, int], obstacles: set) -> bool:
        if pos is None: return False
        dx, dy = self.DIRECTIONS[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        if not (0 <= next_pos[0] < grid_size[0] and 0 <= next_pos[1] < grid_size[1]):
            return False
        if next_pos in obstacles:
            return False
        return True

    def _count_escape_routes(self, pos: Tuple[int, int], grid_size: Tuple[int, int], obstacles: set) -> int:
        count = 0
        for action in [self.UP, self.RIGHT, self.DOWN, self.LEFT]:
            if self._is_valid_move(pos, action, grid_size, obstacles):
                count += 1
        return count

    def calc_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        return self.manhattan_distance(pos, goal)

    def a_star(self, start, goal, obstacles, grid_size):
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start))
        counter += 1
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.calc_heuristic(start, goal)}
        in_open_set = {start}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            in_open_set.discard(current)
            
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            for direction in [self.UP, self.RIGHT, self.DOWN, self.LEFT]:
                dx, dy = self.DIRECTIONS[direction]
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]):
                    continue
                if neighbor in obstacles:
                    continue
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.calc_heuristic(neighbor, goal)
                    if neighbor not in in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        counter += 1
                        in_open_set.add(neighbor)
        return []