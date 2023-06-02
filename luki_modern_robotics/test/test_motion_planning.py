from collections import defaultdict

log_str = ""
def log(string):
    global log_str  # pylint: disable=global-statement
    log_str += string

class OpenList:
    def __init__(self):
        self.dict = {}

    def insert(self, value, est_total_cost):
        self.dict[value] = est_total_cost

    def get_first(self):
        key = None
        cost = None
        for k, v in self.dict.items():
            if not cost or v < cost:
                key = k
                cost = v
        assert key
        del self.dict[key]
        return key

    def empty(self):
        return not bool(self.dict)

    def get_est_total_cost(self, value):
        if not value in self.dict:
            return "oo"
        return self.dict[value]

open_list = OpenList()
closed_list = set()


# costs per edge
cost = defaultdict(lambda: defaultdict(lambda: None))

cost[1][3] = 18
cost[1][4] = 12
cost[1][5] = 30
cost[2][3] = 27
cost[2][6] = 10
cost[3][6] = 15
cost[4][5] = 8
cost[4][6] = 20
cost[5][6] = 10

# not a directed graph so cost[i][j] == cost[j][i]
for i in range(1, 6):
    for j in range(1, 6):
        if i != j:
            if cost[i][j]:
                cost[j][i] = cost[i][j]

# start node
start = 1
# end node
end = 6

def get_neighbors(node):
    result = set()
    for k0, v0 in cost.items():
        for k1, v1 in v0.items():
            if v0 and v1 and node in (k0, k1):
                result.add(k0)
                result.add(k1)
    return result

def get_neighbors_not_in_closed(node):
    nbrs = get_neighbors(node)
    return [n for n in nbrs if n not in closed_list]


# minimum cost so far for node (from start)
past_cost = defaultdict(lambda: 99)

heuristic_cost_to_go = {}
heuristic_cost_to_go[1] = 20
for i in range(2, 6):
    heuristic_cost_to_go[i] = 10
heuristic_cost_to_go[6] = 0

parent = {}


def a_star():
    init_a_star(start)
    while not open_list.empty():
        current = open_list.get_first()
        log(f"\ncurrent: {current}")
        if current == end:
            return end
        closed_list.add(current)
        for nbr in get_neighbors_not_in_closed(current):
            process_neighbor(current, nbr)
    return None


def process_neighbor(current, nbr):
    log(f"\nnbr: {nbr}")
    tentative_past_cost = past_cost[current] + cost[current][nbr]
    if tentative_past_cost < past_cost[nbr]:
        update_best_path(current, nbr, tentative_past_cost)
    print_all()


def update_best_path(current, nbr, tentative_past_cost):
    past_cost[nbr] = tentative_past_cost
    parent[nbr] = current
    est_total_cost = past_cost[nbr] + heuristic_cost_to_go[nbr]
    open_list.insert(nbr, est_total_cost)


def init_a_star(start):
    open_list.insert(start, heuristic_cost_to_go[start])
    past_cost[start] = 0


def print_all():
    s = collect_information()
    log("\n" + s)
    log("="*40)


def collect_information():
    s = print_past_cost()
    s += print_optimist_cost_to_go()
    s += print_est_tot_cost()
    s += print_parent_nodes()
    s += print_open()
    s += print_closed()
    return s


def get_path(item):
    path = []
    nxt = item
    while nxt != start:
        path = [nxt] + path
        nxt = parent[nxt]
    path = [start] + path
    return path


def print_past_cost():
    ret = f"\n{'Past cost:':<15}"

    for i in range(1, 7):
        val = past_cost[i]
        ret += f"{val:>3} | "
    return ret


def print_optimist_cost_to_go():
    ret = f"\n{'optimist ctg:':<15}"

    for i in range(1, 7):
        val = heuristic_cost_to_go[i]
        ret += f'{val:>3} | '
    return ret



def print_est_tot_cost():
    ret = f"\n{'est tot cost':<15}"

    for i in range(1, 7):
        ret += f'{open_list.get_est_total_cost(i):>3} | '
    return ret

def print_parent_nodes():
    ret = f"\n{'parent node:':<15}"

    for i in range(1, 7):
        p = parent.get(i, "-")
        ret += f'{p:>3} | '
    return ret


def print_open():
    ret = "\nOPEN\n"
    lst = list(open_list.dict.items())
    lst.sort(key=lambda x: x[1])
    for l in lst:
        ret += f"{l[0]} ({l[1]}), "
    ret += '\n'
    return ret

def print_closed():
    ret = "CLOSED\n"
    lst = list(closed_list)
    lst.sort()
    for l in lst:
        ret += f"{l}, "
    ret += "\n\n"
    return ret


def test_cost():
    assert not cost[1][6]
    assert not cost[4][2]
    assert cost[4][1] == 12
    assert cost[3][6] == 15
    assert cost[3][2] == 27

def test_init_a_star():
    init_a_star(start)
    assert len(open_list.dict) == 1
    assert open_list.dict[1] == 20
    assert past_cost[1] == 0
    assert len(closed_list) == 0




def test_a_star():
    expected = """
current: 1
nbr: 3

Past cost:       0 |  99 |  18 |  99 |  99 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  oo |  28 |  oo |  oo |  oo | 
parent node:     - |   - |   1 |   - |   - |   - | 
OPEN
3 (28), 
CLOSED
1, 

========================================
nbr: 4

Past cost:       0 |  99 |  18 |  12 |  99 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  oo |  28 |  22 |  oo |  oo | 
parent node:     - |   - |   1 |   1 |   - |   - | 
OPEN
4 (22), 3 (28), 
CLOSED
1, 

========================================
nbr: 5

Past cost:       0 |  99 |  18 |  12 |  30 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  oo |  28 |  22 |  40 |  oo | 
parent node:     - |   - |   1 |   1 |   1 |   - | 
OPEN
4 (22), 3 (28), 5 (40), 
CLOSED
1, 

========================================
current: 4
nbr: 5

Past cost:       0 |  99 |  18 |  12 |  20 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  oo |  28 |  oo |  30 |  oo | 
parent node:     - |   - |   1 |   1 |   4 |   - | 
OPEN
3 (28), 5 (30), 
CLOSED
1, 4, 

========================================
nbr: 6

Past cost:       0 |  99 |  18 |  12 |  20 |  32 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  oo |  28 |  oo |  30 |  32 | 
parent node:     - |   - |   1 |   1 |   4 |   4 | 
OPEN
3 (28), 5 (30), 6 (32), 
CLOSED
1, 4, 

========================================
current: 3
nbr: 2

Past cost:       0 |  45 |  18 |  12 |  20 |  32 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  55 |  oo |  oo |  30 |  32 | 
parent node:     - |   3 |   1 |   1 |   4 |   4 | 
OPEN
5 (30), 6 (32), 2 (55), 
CLOSED
1, 3, 4, 

========================================
nbr: 6

Past cost:       0 |  45 |  18 |  12 |  20 |  32 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  55 |  oo |  oo |  30 |  32 | 
parent node:     - |   3 |   1 |   1 |   4 |   4 | 
OPEN
5 (30), 6 (32), 2 (55), 
CLOSED
1, 3, 4, 

========================================
current: 5
nbr: 6

Past cost:       0 |  45 |  18 |  12 |  20 |  30 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost    oo |  55 |  oo |  oo |  oo |  30 | 
parent node:     - |   3 |   1 |   1 |   4 |   5 | 
OPEN
6 (30), 2 (55), 
CLOSED
1, 3, 4, 5, 

========================================
current: 6"""

    parent = a_star()
    path = get_path(parent)
    assert expected == log_str
    assert path == [1, 4, 5, 6]
