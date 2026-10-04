import heapq
import pygame
import sys


pygame.init()

CELL_SIZE = 30
ROWS = 15
COLS = 15
LINE_WIDTH = 1

SIDEBAR_WIDTH = 200

GRID_WIDTH = COLS * CELL_SIZE
GRID_HEIGHT = ROWS * CELL_SIZE

WIDTH = SIDEBAR_WIDTH + GRID_WIDTH
HEIGHT = GRID_HEIGHT

SIDEBAR_RECT = pygame.Rect(
    0,
    0,
    SIDEBAR_WIDTH,
    HEIGHT
)

TITLE_RECT = pygame.Rect(
    SIDEBAR_RECT.left + 15,
    SIDEBAR_RECT.top + 15,
    SIDEBAR_RECT.width - 30,
    40
)

INSTRUCTIONS_RECT = pygame.Rect(
    SIDEBAR_RECT.left + 15,
    TITLE_RECT.bottom + 20,
    SIDEBAR_RECT.width - 30,
    200
)

GRID_RECT = pygame.Rect(
    SIDEBAR_WIDTH,
    0,
    GRID_WIDTH,
    GRID_HEIGHT
)

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
BLOCKED_COLOR = (255, 255, 255)   # Wall color

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

title_font = pygame.font.SysFont(None, 32)

title_surface = title_font.render(
    "A* Pathfinding",
    True,
    (255, 255, 255)
)

title_text_rect = title_surface.get_rect(
    left=TITLE_RECT.left,
    centery=TITLE_RECT.centery
)

instructions = (
    "Left click to select the start and goal positions. "
    "Right click to add or remove obstacles. "
    "Press SPACE to restart the search and R to reset the grid."
)

font = pygame.font.SysFont(None, 22)


def draw_paragraph(surface, text, font, color, rect, line_spacing=4):
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "

        # Check how wide this line would be
        if font.size(test_line)[0] <= rect.width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "

    # Don't forget the final line
    if current_line:
        lines.append(current_line)

    # Draw each line
    y = rect.top

    for line in lines:
        text_surface = font.render(line.strip(), True, color)
        surface.blit(text_surface, (rect.left, y))

        y += font.get_linesize() + line_spacing

def get_cell_centre(row, col):
    centre_x = GRID_RECT.x + (col * CELL_SIZE) + (CELL_SIZE // 2)
    centre_y = GRID_RECT.y + (row * CELL_SIZE) + (CELL_SIZE // 2)

    return centre_x, centre_y


def heuristic(row_a, col_a, row_b, col_b):
    d_row = abs(row_a - row_b)
    d_col = abs(col_a - col_b)

    diagonal_moves = min(d_row, d_col)
    straight_moves = abs(d_row - d_col)

    return (diagonal_moves * 14) + (straight_moves * 10)


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
    # Return valid neighboring cells (row, col) coordinates
    directions = [
        (-1,  0),  # up
        ( 1,  0),  # down
        ( 0, -1),  # left
        ( 0,  1),  # right
        (-1, -1),  # up-left
        (-1,  1),  # up-right
        ( 1, -1),  # down-left
        ( 1,  1),  # down-right
    ]
    for d_row, d_col in directions:
        n_row = row + d_row
        n_col = col + d_col
        if 0 <= n_row < ROWS and 0 <= n_col < COLS:
            if d_row != 0 and d_col != 0:
                side_cell_1 = grid_matrix[row][n_col]
                side_cell_2 = grid_matrix[n_row][col]

                if not side_cell_1["blocked"] and not side_cell_2["blocked"]:
                    yield n_row, n_col, d_row, d_col

            else:
                yield n_row, n_col, d_row, d_col


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

    for n_row, n_col, d_row, d_col in get_neighbors(current_row, current_col):
        neighbor = grid_matrix[n_row][n_col]

        if neighbor["blocked"] or (n_row, n_col) in closed_set:
            continue

        predecessor = (current_row, current_col)
        if predecessor not in neighbor["history"]:
            neighbor["history"].append(predecessor)

        if d_row != 0 and d_col != 0:
            movement_cost = 14
        else:
            movement_cost = 10
            
        tentative_g = current_cell["g_cost"] + movement_cost
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
            mouse_x, mouse_y = pygame.mouse.get_pos()

            if GRID_RECT.collidepoint(mouse_x, mouse_y):
                grid_x = mouse_x - GRID_RECT.x
                grid_y = mouse_y - GRID_RECT.y

                col = grid_x // CELL_SIZE
                row = grid_y // CELL_SIZE
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

    SCREEN.blit(title_surface, title_text_rect)

    draw_paragraph(
        SCREEN,
        instructions,
        font,
        (255, 255, 255),
        INSTRUCTIONS_RECT
    )

    for row in range(ROWS):
        for col in range(COLS):
            x = GRID_RECT.x + (col * CELL_SIZE)
            y = GRID_RECT.y + (row * CELL_SIZE)
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
    clock.tick(60)

