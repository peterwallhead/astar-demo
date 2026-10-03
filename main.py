import heapq
import pygame
import sys


pygame.init()

CELL_SIZE = 60
ROWS = 15
COLS = 15
LINE_WIDTH = 1

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* Demo")

BG_COLOR = (30, 30, 30)           # Dark charcoal
GRID_COLOR = (70, 70, 70)         # Muted gray
START_COLOR = (0, 180, 120)       # Emerald green
GOAL_COLOR = (180, 0, 120)        # Rose red
OPEN_COLOR = (90, 180, 255)       # Open set (frontier)
CLOSED_COLOR = (80, 120, 220)     # Closed set
PATH_COLOR = (245, 200, 66)       # Final selected path
CURRENT_COLOR = (255, 165, 0)     # Currently expanded node
BLOCKED_COLOR = (20, 20, 20)      # Wall color

start_set = False
goal_set = False
start_pos = None
goal_pos = None

search_running = False
search_finished = False

open_heap = []
push_id = 0
closed_set = set()
current_pos = None

grid_matrix = [
    [
        {
            "blocked": False,
            "start": False,
            "goal": False,
            "open": False,
            "closed": False,
            "in_path": False,
            "g_cost": float("inf"),
            "h_cost": float("inf"),
            "f_cost": float("inf"),
            "parent": None,
            "history": [],
        }
        for _ in range(COLS)
    ]
    for _ in range(ROWS)
]


def get_cell_centre(row, col):
    centre_x = (col * CELL_SIZE) + (CELL_SIZE // 2)
    centre_y = (row * CELL_SIZE) + (CELL_SIZE // 2)
    return centre_x, centre_y


def heuristic(row_a, col_a, row_b, col_b):
    return abs(row_a - row_b) + abs(col_a - col_b)


def reset_search_state():
    global open_heap, push_id, closed_set, search_running, search_finished, current_pos

    open_heap = []
    push_id = 0
    closed_set = set()
    search_running = False
    search_finished = False
    current_pos = None

    for row in range(ROWS):
        for col in range(COLS):
            cell = grid_matrix[row][col]
            cell["open"] = False
            cell["closed"] = False
            cell["in_path"] = False
            cell["g_cost"] = float("inf")
            cell["h_cost"] = float("inf")
            cell["f_cost"] = float("inf")
            cell["parent"] = None
            cell["history"] = []


def push_open_cell(row, col):
    global push_id
    cell = grid_matrix[row][col]
    heapq.heappush(open_heap, (cell["f_cost"], cell["g_cost"], push_id, row, col))
    cell["open"] = True
    push_id += 1


def pop_best_open_cell():
    while open_heap:
        f_cost, g_cost, _, row, col = heapq.heappop(open_heap)
        cell = grid_matrix[row][col]

        if (row, col) in closed_set:
            continue

        if f_cost != cell["f_cost"] or g_cost != cell["g_cost"]:
            continue

        cell["open"] = False
        return row, col

    return None


def build_selected_path(goal_row, goal_col):
    current = (goal_row, goal_col)
    while current is not None:
        row, col = current
        cell = grid_matrix[row][col]
        if not cell["start"] and not cell["goal"]:
            cell["in_path"] = True
        current = cell["parent"]


def start_search():
    global search_running

    if not start_set or not goal_set:
        return

    reset_search_state()

    start_row, start_col = start_pos
    goal_row, goal_col = goal_pos

    start_cell = grid_matrix[start_row][start_col]
    start_cell["g_cost"] = 0
    start_cell["h_cost"] = heuristic(start_row, start_col, goal_row, goal_col)
    start_cell["f_cost"] = start_cell["g_cost"] + start_cell["h_cost"]

    push_open_cell(start_row, start_col)
    search_running = True


def get_neighbors(row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d_row, d_col in directions:
        n_row = row + d_row
        n_col = col + d_col
        if 0 <= n_row < ROWS and 0 <= n_col < COLS:
            yield n_row, n_col


def a_star_step():
    global search_running, search_finished, current_pos

    best = pop_best_open_cell()
    if best is None:
        current_pos = None
        search_running = False
        search_finished = True
        print("No path found.")
        return

    current_row, current_col = best
    current_pos = (current_row, current_col)
    current_cell = grid_matrix[current_row][current_col]

    if current_cell["goal"]:
        build_selected_path(current_row, current_col)
        search_running = False
        search_finished = True
        print("Goal reached. Search stopped.")
        return

    closed_set.add((current_row, current_col))
    if not current_cell["start"] and not current_cell["goal"]:
        current_cell["closed"] = True

    for n_row, n_col in get_neighbors(current_row, current_col):
        neighbor = grid_matrix[n_row][n_col]

        if neighbor["blocked"] or (n_row, n_col) in closed_set:
            continue

        predecessor = (current_row, current_col)
        if predecessor not in neighbor["history"]:
            neighbor["history"].append(predecessor)

        tentative_g = current_cell["g_cost"] + 1
        if tentative_g < neighbor["g_cost"]:
            goal_row, goal_col = goal_pos
            neighbor["parent"] = predecessor
            neighbor["g_cost"] = tentative_g
            neighbor["h_cost"] = heuristic(n_row, n_col, goal_row, goal_col)
            neighbor["f_cost"] = neighbor["g_cost"] + neighbor["h_cost"]
            push_open_cell(n_row, n_col)


clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            col = mouse_pos[0] // CELL_SIZE
            row = mouse_pos[1] // CELL_SIZE

            if 0 <= row < ROWS and 0 <= col < COLS:
                clicked_cell = grid_matrix[row][col]

                if event.button == 1:  # Left click
                    if not clicked_cell["start"] and not clicked_cell["goal"] and not start_set:
                        clicked_cell["start"] = True
                        start_set = True
                        start_pos = (row, col)
                        print(f"Start Cell Set at ({row}, {col})")

                    elif not clicked_cell["start"] and not clicked_cell["goal"] and not goal_set:
                        clicked_cell["goal"] = True
                        goal_set = True
                        goal_pos = (row, col)
                        print(f"Goal Cell Set at ({row}, {col})")
                        start_search()

                elif event.button == 3:  # Right click: toggle blocked
                    if not clicked_cell["start"] and not clicked_cell["goal"]:
                        clicked_cell["blocked"] = not clicked_cell["blocked"]

                centre_x, centre_y = get_cell_centre(row, col)
                print(f"Clicked Cell [Row {row}, Col {col}] -> Center Pixel: ({centre_x}, {centre_y})")

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                start_set = False
                goal_set = False
                start_pos = None
                goal_pos = None
                reset_search_state()

                for row in range(ROWS):
                    for col in range(COLS):
                        cell = grid_matrix[row][col]
                        cell["start"] = False
                        cell["goal"] = False
                        cell["blocked"] = False

            elif event.key == pygame.K_SPACE:
                if start_set and goal_set:
                    start_search()

    if search_running:
        a_star_step()

    SCREEN.fill(BG_COLOR)

    for row in range(ROWS):
        for col in range(COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            cell = grid_matrix[row][col]

            if cell["start"]:
                pygame.draw.rect(SCREEN, START_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell["goal"]:
                pygame.draw.rect(SCREEN, GOAL_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell["blocked"]:
                pygame.draw.rect(SCREEN, BLOCKED_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            elif current_pos == (row, col) and search_running:
                pygame.draw.rect(SCREEN, CURRENT_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell["in_path"]:
                pygame.draw.rect(SCREEN, PATH_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell["closed"]:
                pygame.draw.rect(SCREEN, CLOSED_COLOR, (x, y, CELL_SIZE, CELL_SIZE))
            elif cell["open"]:
                pygame.draw.rect(SCREEN, OPEN_COLOR, (x, y, CELL_SIZE, CELL_SIZE))

            pygame.draw.rect(SCREEN, GRID_COLOR, (x, y, CELL_SIZE, CELL_SIZE), LINE_WIDTH)

    pygame.display.flip()
    clock.tick(120)