import pygame
import random
from collections import deque
import time

# Konfigurasi
GEN_DELAY = 10         # Delay generasi labirin (ms)
SOLVE_DELAY = 130      # Delay solusi (ms)
FINAL_DELAY = 10000    # Delay akhir (ms)
ROWS_SIZE = 15         # Ukuran panjang
COLS_SIZE = 15         # Ukuran lebar
CELL_SIZE = 30         # Ukuran cell
TREE_WIDTH = 400       # Lebar area tree
VERTICAL_SPACING = 50  # Jarak vertikal antara node tree
NODE_RADIUS = 8        # Radius node tree
ENABLE_PAUSE = True    # Aktifkan fitur pause

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False

class TreeNode:
    def __init__(self, pos):
        self.pos = pos  # (x, y)
        self.children = []
        self.parent = None

def generate_maze(rows, cols):
    grid = [[Cell(i, j) for j in range(cols)] for i in range(rows)]
    stack = []
    current = grid[0][0]
    current.visited = True
    stack.append(current)
    
    directions = [
        ('top', -1, 0),
        ('right', 0, 1),
        ('bottom', 1, 0),
        ('left', 0, -1)
    ]
    
    while stack:
        neighbors = []
        for direction, dx, dy in directions:
            nx = current.x + dx
            ny = current.y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbor = grid[nx][ny]
                if not neighbor.visited:
                    neighbors.append((direction, neighbor))
        
        if neighbors:
            direction, next_cell = random.choice(neighbors)
            if direction == 'top':
                current.walls['top'] = False
                next_cell.walls['bottom'] = False
            elif direction == 'right':
                current.walls['right'] = False
                next_cell.walls['left'] = False
            elif direction == 'bottom':
                current.walls['bottom'] = False
                next_cell.walls['top'] = False
            elif direction == 'left':
                current.walls['left'] = False
                next_cell.walls['right'] = False
            
            next_cell.visited = True
            stack.append(current)
            current = next_cell
            yield grid, current
        else:
            current = stack.pop()
            yield grid, current
    
    yield grid, None

def maze_to_tree(grid, start_pos):
    root = TreeNode(start_pos)
    visited = set([start_pos])
    stack = [(root, start_pos)]
    
    directions = [
        ('top', -1, 0),
        ('right', 0, 1),
        ('bottom', 1, 0),
        ('left', 0, -1)
    ]
    
    while stack:
        node, (x, y) = stack.pop()
        cell = grid[x][y]
        
        for dir_name, dx, dy in directions:
            if not cell.walls[dir_name]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    child = TreeNode((nx, ny))
                    child.parent = node
                    node.children.append(child)
                    stack.append((child, (nx, ny)))
    return root

def calculate_tree_stats(root):
    if not root:
        return 0, 0
    
    max_depth = 0
    max_width = 0
    current_level = [root]
    
    while current_level:
        max_width = max(max_width, len(current_level))
        next_level = []
        for node in current_level:
            next_level.extend(node.children)
        
        current_level = next_level
        if current_level:
            max_depth += 1
    
    return max_depth, max_width

def draw_maze(screen, grid, cell_size, current=None):
    screen.fill((255, 255, 255))
    for row in grid:
        for cell in row:
            x = cell.y * cell_size
            y = cell.x * cell_size
            if cell.walls['top']:
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x + cell_size, y), 2)
            if cell.walls['right']:
                pygame.draw.line(screen, (0, 0, 0), (x + cell_size, y), (x + cell_size, y + cell_size), 2)
            if cell.walls['bottom']:
                pygame.draw.line(screen, (0, 0, 0), (x, y + cell_size), (x + cell_size, y + cell_size), 2)
            if cell.walls['left']:
                pygame.draw.line(screen, (0, 0, 0), (x, y), (x, y + cell_size), 2)
    
    if current:
        pygame.draw.rect(screen, (255, 0, 0), 
                         (current.y * cell_size + 2, current.x * cell_size + 2, 
                          cell_size - 4, cell_size - 4))

def solve_bfs(tree_root, end_pos):
    queue = deque([tree_root])
    steps = 0

    while queue:
        steps += 1
        node = queue.popleft()

        if node.pos == end_pos:
            path = []
            while node:
                path.append(node.pos)
                node = node.parent
            path.reverse()
            yield path[0], [], True, steps
            return path, steps
        
        new_nodes = []
        for child in node.children:
            queue.append(child)
            new_nodes.append(child.pos)
        
        yield node.pos, new_nodes, False, steps

def solve_dfs(tree_root, end_pos):
    stack = [tree_root]
    steps = 0

    while stack:
        steps += 1
        node = stack.pop()

        if node.pos == end_pos:
            path = []
            while node:
                path.append(node.pos)
                node = node.parent
            path.reverse()
            yield path[0], [], True, steps
            return path, steps
        
        new_nodes = []
        for child in reversed(node.children):
            stack.append(child)
            new_nodes.append(child.pos)

        yield node.pos, new_nodes, False, steps

def draw_solving_progress(screen, cell_size, current, path=[], history=None, color=(0, 255, 0), paused=False):
    if history is None:
        history = []
    
    if current:
        x = current[1] * cell_size
        y = current[0] * cell_size
        pygame.draw.rect(screen, (200, 200, 255), 
                         (x + 2, y + 2, cell_size - 4, cell_size - 4))
        
        if not paused and (not history or history[-1][0] != current):
            history.append((current, pygame.time.get_ticks()))
    
    for pos, timestamp in history:
        alpha = 255 if paused else max(0, 255 - (pygame.time.get_ticks() - timestamp) // (SOLVE_DELAY * 2))
        if alpha > 0:
            x = pos[1] * cell_size + cell_size // 2
            y = pos[0] * cell_size + cell_size // 2
            radius = cell_size // 3
            
            surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            pygame.draw.circle(surf, (200, 200, 255, alpha), (radius, radius), radius)
            screen.blit(surf, (x - radius, y - radius))

    if len(path) > 1:
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            start_x = y1 * cell_size + cell_size // 2
            start_y = x1 * cell_size + cell_size // 2
            end_x = y2 * cell_size + cell_size // 2
            end_y = x2 * cell_size + cell_size // 2
            pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), 3)

def draw_path(screen, path, cell_size, color, alpha):
    if len(path) < 2:
        return
    
    temp_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        start_x = y1 * cell_size + cell_size // 2
        start_y = x1 * cell_size + cell_size // 2
        end_x = y2 * cell_size + cell_size // 2
        end_y = x2 * cell_size + cell_size // 2
        pygame.draw.line(temp_surface, (*color, alpha), (start_x, start_y), (end_x, end_y), 3)
    screen.blit(temp_surface, (0, 0))

def compute_tree_layout(level_nodes, tree_width, vertical_spacing):
    node_positions = {}
    for depth, nodes in level_nodes.items():
        nodes_sorted = sorted(nodes, key=lambda node: (node[0], node[1]))
        n = len(nodes_sorted)
        for idx, node in enumerate(nodes_sorted):
            x = (idx + 1) * (tree_width / (n + 1))
            y = 20 + depth * vertical_spacing
            node_positions[node] = (x, y)
    return node_positions

def draw_tree(surface, node_positions, parent_tree, current_node, start_node, end_node, tree_offset_y=0):
    for node, pos in node_positions.items():
        if node == start_node:
            continue
        parent = parent_tree.get(node)
        if parent is not None and parent in node_positions:
            parent_pos = node_positions[parent]
            adjusted_parent_pos = (parent_pos[0], parent_pos[1] - tree_offset_y)
            adjusted_pos = (pos[0], pos[1] - tree_offset_y)
            pygame.draw.line(surface, (150, 150, 150), adjusted_parent_pos, adjusted_pos, 2)

    font = pygame.font.SysFont(None, 18)
    for node, pos in node_positions.items():
        adjusted_pos = (pos[0], pos[1] - tree_offset_y)
        
        if node == start_node:
            color = (0, 0, 255)   # Biru untuk start
        elif node == end_node:
            color = (255, 0, 0)    # Merah untuk end
        elif node == current_node:
            color = (255, 255, 0)  # Kuning untuk node aktif
        else:
            color = (100, 100, 100) # Abu-abu untuk node biasa
        
        pygame.draw.circle(surface, color, (int(adjusted_pos[0]), int(adjusted_pos[1])), NODE_RADIUS)
        coord_text = f"({node[1]},{node[0]})"
        text_surf = font.render(coord_text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=(int(adjusted_pos[0]), int(adjusted_pos[1]) - 15))
        surface.blit(text_surf, text_rect)

def main():
    maze_width = COLS_SIZE * CELL_SIZE
    maze_height = ROWS_SIZE * CELL_SIZE
    total_width = maze_width + TREE_WIDTH
    total_height = maze_height
    
    pygame.init()
    screen = pygame.display.set_mode((total_width, total_height))
    pygame.display.set_caption("Perbandingan Komputasi BFS dan DFS dalam Tree Maze")
    
    maze_gen = generate_maze(ROWS_SIZE, COLS_SIZE)
    solving_bfs = None
    solving_dfs = None
    bfs_path = []
    dfs_path = []
    solving_history_bfs = []
    solving_history_dfs = []
    state = "generating"
    
    # Struktur untuk tree visualization
    parent_tree = {}
    depth_tree = {}
    level_nodes = {}
    node_positions = {}
    
    # Variabel untuk scrolling tree
    tree_offset_y = 0
    scroll_speed = 20
    
    # Variabel untuk pause
    paused = False
    
    # Statistik
    tree_depth = 0
    tree_width = 0
    bfs_steps = 0
    dfs_steps = 0
    bfs_time = 0
    dfs_time = 0
    # Font
    font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 36)
    
    running = True
    while running:
        pygame.K_PAUSE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and ENABLE_PAUSE and state in ["solving_bfs", "solving_dfs", "paused_bfs", "paused_dfs"]:
                    paused = not paused
                
                elif state == "paused_bfs" and event.key:
                    # Reset struktur tree untuk DFS
                    tree_offset_y = 0
                    parent_tree = {}
                    depth_tree = {}
                    level_nodes = {}
                    start = grid[0][0]
                    end = grid[-1][-1]
                    depth_tree[(start.x, start.y)] = 0
                    level_nodes[0] = [(start.x, start.y)]
                    node_positions = compute_tree_layout(level_nodes, TREE_WIDTH, VERTICAL_SPACING)
                    tree_root = maze_to_tree(grid, (start.x, start.y))
                    solving_dfs = solve_dfs(tree_root, (end.x, end.y))
                    state = "solving_dfs"
                    paused = False
                    start_time_dfs = time.time()
                
                elif state == "paused_dfs" and event.key:
                    state = "show_final"
                    paused = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    tree_offset_y = max(0, tree_offset_y - scroll_speed)
                elif event.button == 5:  # Scroll down
                    tree_offset_y += scroll_speed
        
        if paused:
            screen.fill((255, 255, 255))
            draw_maze(screen, grid, CELL_SIZE)

            if state == "solving_bfs":
                draw_solving_progress(screen, CELL_SIZE, current_pos, bfs_path, solving_history_bfs, (0, 255, 0), True)
            elif state == "solving_dfs":
                draw_solving_progress(screen, CELL_SIZE, current_pos, dfs_path, solving_history_dfs, (255, 0, 0), True)

            tree_surface = pygame.Surface((TREE_WIDTH, total_height))
            tree_surface.fill((255, 255, 255))
            draw_tree(tree_surface, node_positions, parent_tree, None, 
                    (start.x, start.y), (end.x, end.y), tree_offset_y)
            screen.blit(tree_surface, (maze_width, 0))

            overlay = pygame.Surface((total_width, total_height), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, 100))
            screen.blit(overlay, (0, 0))

            pause_text = title_font.render("PAUSED - Press SPACE to continue", True, (0, 0, 0))
            screen.blit(pause_text, (total_width // 2 - pause_text.get_width() // 2, 30))
            
            pygame.display.update()
            continue
        
        if state == "generating":
            try:

                grid, current = next(maze_gen)
                screen.fill((255, 255, 255))
                draw_maze(screen, grid, CELL_SIZE, current)
                pygame.display.update()
                pygame.time.wait(GEN_DELAY)
            except StopIteration:
                state = "solving_bfs"
                start = grid[0][0]
                end = grid[-1][-1]
                
                # Bangun representasi tree dari maze
                start_pos = (start.x, start.y)
                tree_root = maze_to_tree(grid, start_pos)
                tree_depth, tree_width = calculate_tree_stats(tree_root)
                solving_bfs = solve_bfs(tree_root, (end.x, end.y))

                # Inisialisasi struktur tree untuk BFS
                parent_tree = {}
                depth_tree = {(start.x, start.y): 0}
                level_nodes = {0: [(start.x, start.y)]}
                node_positions = compute_tree_layout(level_nodes, TREE_WIDTH, VERTICAL_SPACING)
                
                # Mulai pengukuran waktu BFS
                start_time_bfs = time.time()
        
        elif state == "solving_bfs":
            try:
                result, new_nodes, is_final, steps = next(solving_bfs)
                bfs_steps = steps
                if is_final:
                    try:
                        current_pos = (end.x, end.y)
                        new_nodes_list = []
                        next(solving_bfs)  # Skip final step
                    except StopIteration as e:
                        bfs_path = e.value[0]
                        bfs_time = time.time() - start_time_bfs
                        state = "paused_bfs" if ENABLE_PAUSE else "solving_dfs"
                        
                        if not ENABLE_PAUSE:
                            # Reset struktur tree untuk DFS
                            parent_tree = {}
                            depth_tree = {(start.x, start.y): 0}
                            level_nodes = {0: [(start.x, start.y)]}
                            node_positions = compute_tree_layout(level_nodes, TREE_WIDTH, VERTICAL_SPACING)
                            tree_root = maze_to_tree(grid, start_pos)
                            solving_dfs = solve_dfs(tree_root, (end.x, end.y))
                            start_time_dfs = time.time()
                else:
                    current_pos, new_nodes_list = result, new_nodes
                    
                    # Update struktur tree dengan node baru
                    for node in new_nodes_list:
                        parent_tree[node] = current_pos
                        depth_tree[node] = depth_tree[parent_tree[node]] + 1
                        d = depth_tree[node]
                        if d not in level_nodes:
                            level_nodes[d] = []
                        level_nodes[d].append(node)
                    
                    # Hitung ulang layout tree
                    node_positions = compute_tree_layout(level_nodes, TREE_WIDTH, VERTICAL_SPACING)
                    
                    # Gambar maze dan tree
                    screen.fill((255, 255, 255))
                    draw_maze(screen, grid, CELL_SIZE)
                    draw_solving_progress(screen, CELL_SIZE, current_pos, bfs_path, solving_history_bfs, (0, 255, 0))
                    
                    # Gambar tree di sebelah kanan
                    tree_surface = pygame.Surface((TREE_WIDTH, total_height))
                    tree_surface.fill((255, 255, 255))
                    draw_tree(tree_surface, node_positions, parent_tree, current_pos, 
                             (start.x, start.y), (end.x, end.y), tree_offset_y)
                    screen.blit(tree_surface, (maze_width, 0))
                    
                    # Tampilkan statistik BFS
                    stats_text = font.render(f"BFS Steps: {bfs_steps}", True, (0, 0, 0))
                    screen.blit(stats_text, (10, 10))
                    
                    pygame.display.update()
                    pygame.time.wait(SOLVE_DELAY)
            except StopIteration:
                pass
        
        elif state == "paused_bfs":
            screen.fill((255, 255, 255))
            draw_maze(screen, grid, CELL_SIZE)
            draw_solving_progress(screen, CELL_SIZE, current_pos, bfs_path, solving_history_bfs, (0, 255, 0), True)
            
            # Gambar tree di sebelah kanan
            tree_surface = pygame.Surface((TREE_WIDTH, total_height))
            tree_surface.fill((255, 255, 255))
            draw_tree(tree_surface, node_positions, parent_tree, None, 
                     (start.x, start.y), (end.x, end.y), tree_offset_y)
            screen.blit(tree_surface, (maze_width, 0))
            
            # Tampilkan statistik BFS
            stats_text = font.render(f"BFS Completed! Steps: {bfs_steps}, Time: {bfs_time:.4f}s", True, (0, 0, 0))
            screen.blit(stats_text, (10, 10))
            
            pause_text = title_font.render("BFS Completed - Press any key to continue", True, (0, 0, 0))
            screen.blit(pause_text, (total_width // 2 - pause_text.get_width() // 2, 100))
            
            pygame.display.update()
        
        elif state == "solving_dfs":
            try:
                result, new_nodes, is_final, steps = next(solving_dfs)
                dfs_steps = steps
                if is_final:
                    try:
                        current_pos = (end.x, end.y)
                        new_nodes_list = []
                        next(solving_dfs)  # Skip final step
                    except StopIteration as e:
                        dfs_path = e.value[0]
                        dfs_time = time.time() - start_time_dfs
                        state = "paused_dfs" if ENABLE_PAUSE else "show_final"
                else:
                    current_pos, new_nodes_list = result, new_nodes
                    
                    # Update struktur tree dengan node baru
                    for node in new_nodes_list:
                        parent_tree[node] = current_pos
                        depth_tree[node] = depth_tree[parent_tree[node]] + 1
                        d = depth_tree[node]
                        if d not in level_nodes:
                            level_nodes[d] = []
                        level_nodes[d].append(node)
                    
                    # Hitung ulang layout tree
                    node_positions = compute_tree_layout(level_nodes, TREE_WIDTH, VERTICAL_SPACING)
                    
                    # Gambar maze dan tree
                    screen.fill((255, 255, 255))
                    draw_maze(screen, grid, CELL_SIZE)
                    draw_solving_progress(screen, CELL_SIZE, current_pos, dfs_path, solving_history_dfs, (255, 0, 0))
                    
                    # Gambar tree di sebelah kanan
                    tree_surface = pygame.Surface((TREE_WIDTH, total_height))
                    tree_surface.fill((255, 255, 255))
                    draw_tree(tree_surface, node_positions, parent_tree, current_pos, 
                             (start.x, start.y), (end.x, end.y), tree_offset_y)
                    screen.blit(tree_surface, (maze_width, 0))
                    
                    # Tampilkan statistik DFS
                    stats_text = font.render(f"DFS Steps: {dfs_steps}", True, (0, 0, 0))
                    screen.blit(stats_text, (10, 40))
                    
                    pygame.display.update()
                    pygame.time.wait(SOLVE_DELAY)
            except StopIteration:
                pass
        
        elif state == "paused_dfs":
            screen.fill((255, 255, 255))
            draw_maze(screen, grid, CELL_SIZE)
            draw_solving_progress(screen, CELL_SIZE, current_pos, dfs_path, solving_history_dfs, (255, 0, 0), True)
            
            # Gambar tree di sebelah kanan
            tree_surface = pygame.Surface((TREE_WIDTH, total_height))
            tree_surface.fill((255, 255, 255))
            draw_tree(tree_surface, node_positions, parent_tree, None, 
                     (start.x, start.y), (end.x, end.y), tree_offset_y)
            screen.blit(tree_surface, (maze_width, 0))
            
            # Tampilkan statistik DFS
            stats_text = font.render(f"DFS Completed! Steps: {dfs_steps}, Time: {dfs_time:.4f}s", True, (0, 0, 0))
            screen.blit(stats_text, (10, 40))
            
            pause_text = title_font.render("DFS Completed - Press any key to continue", True, (0, 0, 0))
            screen.blit(pause_text, (total_width // 2 - pause_text.get_width() // 2, 100))
            
            pygame.display.update()
        
        elif state == "show_final":
            screen.fill((255, 255, 255))
            draw_maze(screen, grid, CELL_SIZE)
            draw_path(screen, bfs_path, CELL_SIZE, (255, 255, 0), 180)
            
            # Tampilkan statistik lengkap
            stats_surface = pygame.Surface((TREE_WIDTH, total_height))
            stats_surface.fill((240, 240, 240))
            
            title = title_font.render("Perbandingan Komputasi", True, (0, 0, 0))
            stats_surface.blit(title, (20, 20))
            
            # Statistik tree
            tree_title = font.render("Struktur Tree Maze:", True, (0, 0, 0))
            stats_surface.blit(tree_title, (20, 70))
            depth_text = font.render(f"Kedalaman Tree: {tree_depth}", True, (0, 0, 0))
            stats_surface.blit(depth_text, (40, 100))
            width_text = font.render(f"Lebar Tree: {tree_width}", True, (0, 0, 0))
            stats_surface.blit(width_text, (40, 130))
            
            # Statistik BFS
            bfs_title = font.render("BFS:", True, (0, 100, 0))
            stats_surface.blit(bfs_title, (20, 180))
            bfs_step_text = font.render(f"Langkah: {bfs_steps}", True, (0, 0, 0))
            stats_surface.blit(bfs_step_text, (40, 210))
            bfs_time_text = font.render(f"Waktu: {bfs_time:.4f} detik", True, (0, 0, 0))
            stats_surface.blit(bfs_time_text, (40, 240))
            bfs_path_text = font.render(f"Panjang Jalur: {len(bfs_path)}", True, (0, 0, 0))
            stats_surface.blit(bfs_path_text, (40, 270))
            
            # Statistik DFS
            dfs_title = font.render("DFS:", True, (200, 0, 0))
            stats_surface.blit(dfs_title, (20, 320))
            dfs_step_text = font.render(f"Langkah: {dfs_steps}", True, (0, 0, 0))
            stats_surface.blit(dfs_step_text, (40, 350))
            dfs_time_text = font.render(f"Waktu: {dfs_time:.4f} detik", True, (0, 0, 0))
            stats_surface.blit(dfs_time_text, (40, 380))
            dfs_path_text = font.render(f"Panjang Jalur: {len(dfs_path)}", True, (0, 0, 0))
            stats_surface.blit(dfs_path_text, (40, 410))
            
            # Kesimpulan
            conclusion_title = font.render("Kesimpulan:", True, (0, 0, 0))
            stats_surface.blit(conclusion_title, (20, 460))
            
            if bfs_steps < dfs_steps:
                conclusion = "BFS lebih efisien dalam jumlah langkah"
            elif dfs_steps < bfs_steps:
                conclusion = "DFS lebih efisien dalam jumlah langkah"
            else:
                conclusion = "BFS dan DFS sama efisiennya"
                
            conc_text = font.render(conclusion, True, (0, 0, 0))
            stats_surface.blit(conc_text, (40, 490))
            
            if len(bfs_path) < len(dfs_path):
                path_conc = "BFS menemukan jalur terpendek"
            elif len(dfs_path) < len(bfs_path):
                path_conc = "DFS menemukan jalur lebih pendek"
            else:
                path_conc = "Kedua algoritma menemukan jalur sama panjang"
                
            path_text = font.render(path_conc, True, (0, 0, 0))
            stats_surface.blit(path_text, (40, 520))
            
            screen.blit(stats_surface, (maze_width, 0))
            pygame.display.update()
        else:
            running = False
    
    pygame.quit()

if __name__ == "__main__":
    main()