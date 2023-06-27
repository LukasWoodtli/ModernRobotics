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

class AStar:  # pylint: disable=too-many-instance-attributes

    def __init__(self, start, end, cost_list, heuristic_cost_to_go):
        self.open_list = OpenList()
        self.closed_list = set()
        self.cost = self.init_cost(cost_list)
        self.heuristic_cost_to_go = self.init_heuristic_cost_to_go(heuristic_cost_to_go)
        self.start = start
        self.end = end
        # minimum cost so far for node (from start)
        self.past_cost = defaultdict(lambda: 99)
        self.parent = {}
        self.init_a_star()

    def init_a_star(self):
        self.open_list.insert(self.start, self.heuristic_cost_to_go[self.start])
        self.past_cost[self.start] = 0

    @staticmethod
    def init_cost(cost_list):
        # costs per edge
        res = defaultdict(lambda: defaultdict(lambda: None))
        for cost in cost_list:
            start, end, cst = cost
            res[start][end] = cst
        # not a directed graph so cost[i][j] == cost[j][i]
        for i in range(1, 6):
            for j in range(1, 6):
                if i != j:
                    if res[i][j]:
                        res[j][i] = res[i][j]
        return res

    @staticmethod
    def init_heuristic_cost_to_go(htg):
        heuristic_cost_to_go = {}
        for i in htg:
            node, cost = i
            heuristic_cost_to_go[node] = cost
        return heuristic_cost_to_go

    def get_neighbors(self, node):
        result = set()
        for k0, v0 in self.cost.items():
            for k1, v1 in v0.items():
                if v0 and v1 and node in (k0, k1):
                    result.add(k0)
                    result.add(k1)
        return result

    def get_neighbors_not_in_closed(self, node):
        nbrs = self.get_neighbors(node)
        return [n for n in nbrs if n not in self.closed_list]

    def a_star(self):
        while not self.open_list.empty():
            current = self.open_list.get_first()
            log(f"\ncurrent: {current}")
            if current == self.end:
                return self.get_path(self.end)
            self.closed_list.add(current)
            for nbr in self.get_neighbors_not_in_closed(current):
                self.process_neighbor(current, nbr)
        return None

    def process_neighbor(self, current, nbr):
        log(f"\nnbr: {nbr}")
        tentative_past_cost = self.past_cost[current] + self.cost[current][nbr]
        if tentative_past_cost < self.past_cost[nbr]:
            self.update_best_path(current, nbr, tentative_past_cost)
        self.print_all()


    def update_best_path(self, current, nbr, tentative_past_cost):
        self.past_cost[nbr] = tentative_past_cost
        self.parent[nbr] = current
        est_total_cost = self.past_cost[nbr] + self.heuristic_cost_to_go[nbr]
        self.open_list.insert(nbr, est_total_cost)

    def print_all(self):
        s = self.collect_information()
        log("\n" + s)
        log("="*40)

    @staticmethod
    def print_entry_for_nodes(row_title, list_entries):
        ret = f"\n{f'{row_title}:':<15}"
        for e in range(1, 7):
            ret += f"{list_entries[e]:>3} | "
        return ret

    def collect_information(self):
        s = self.print_entry_for_nodes("Past cost", self.past_cost)

        s += self.print_entry_for_nodes("optimist ctg", self.heuristic_cost_to_go)

        lst = {i: self.open_list.get_est_total_cost(i) for i in range(1, 7)}
        s += self.print_entry_for_nodes("est tot cost", lst)

        lst1 = {i1: self.parent.get(i1, "-") for i1 in range(1, 7)}
        s += self.print_entry_for_nodes('parent node', lst1)

        s += self.print_open()
        s += self.print_closed()
        return s

    def get_path(self, item):
        path = []
        nxt = item
        while nxt != self.start:
            path = [nxt] + path
            nxt = self.parent[nxt]
        path = [self.start] + path
        return path

    def print_open(self):
        ret = "\nOPEN\n"
        lst = list(self.open_list.dict.items())
        lst.sort(key=lambda x: x[1])
        for l in lst:
            ret += f"{l[0]} ({l[1]}), "
        ret += '\n'
        return ret

    def print_closed(self):
        ret = "CLOSED\n"
        lst = list(self.closed_list)
        lst.sort()
        for l in lst:
            ret += f"{l}, "
        ret += "\n\n"
        return ret


test_cost_list = [
    [1, 3, 18],
    [1, 4, 12],
    [1, 5, 30],
    [2, 3, 27],
    [2, 6, 10],
    [3, 6, 15],
    [4, 5, 8],
    [4, 6, 20],
    [5, 6, 10]
]

test_heuristic_cost_to_go = [
    [1, 20],
    [2, 10],
    [3, 10],
    [4, 10],
    [5, 10],
    [6, 0],
]

def test_cost():
    a_star = AStar(1, 6, test_cost_list, test_heuristic_cost_to_go)
    assert not a_star.cost[1][6]
    assert not a_star.cost[4][2]
    assert a_star.cost[4][1] == 12
    assert a_star.cost[3][6] == 15
    assert a_star.cost[3][2] == 27

def test_init_a_star():
    a_star = AStar(1, 6, test_cost_list, test_heuristic_cost_to_go)
    assert len(a_star.open_list.dict) == 1
    assert a_star.open_list.dict[1] == 20
    assert a_star.past_cost[1] == 0
    assert len(a_star.closed_list) == 0




def test_a_star():
    expected = """
current: 1
nbr: 3

Past cost:       0 |  99 |  18 |  99 |  99 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost:   oo |  oo |  28 |  oo |  oo |  oo | 
parent node:     - |   - |   1 |   - |   - |   - | 
OPEN
3 (28), 
CLOSED
1, 

========================================
nbr: 4

Past cost:       0 |  99 |  18 |  12 |  99 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost:   oo |  oo |  28 |  22 |  oo |  oo | 
parent node:     - |   - |   1 |   1 |   - |   - | 
OPEN
4 (22), 3 (28), 
CLOSED
1, 

========================================
nbr: 5

Past cost:       0 |  99 |  18 |  12 |  30 |  99 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost:   oo |  oo |  28 |  22 |  40 |  oo | 
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
est tot cost:   oo |  oo |  28 |  oo |  30 |  oo | 
parent node:     - |   - |   1 |   1 |   4 |   - | 
OPEN
3 (28), 5 (30), 
CLOSED
1, 4, 

========================================
nbr: 6

Past cost:       0 |  99 |  18 |  12 |  20 |  32 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost:   oo |  oo |  28 |  oo |  30 |  32 | 
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
est tot cost:   oo |  55 |  oo |  oo |  30 |  32 | 
parent node:     - |   3 |   1 |   1 |   4 |   4 | 
OPEN
5 (30), 6 (32), 2 (55), 
CLOSED
1, 3, 4, 

========================================
nbr: 6

Past cost:       0 |  45 |  18 |  12 |  20 |  32 | 
optimist ctg:   20 |  10 |  10 |  10 |  10 |   0 | 
est tot cost:   oo |  55 |  oo |  oo |  30 |  32 | 
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
est tot cost:   oo |  55 |  oo |  oo |  oo |  30 | 
parent node:     - |   3 |   1 |   1 |   4 |   5 | 
OPEN
6 (30), 2 (55), 
CLOSED
1, 3, 4, 5, 

========================================
current: 6"""

    a_star = AStar(1, 6, test_cost_list, test_heuristic_cost_to_go)
    path = a_star.a_star()
    assert expected == log_str
    assert path == [1, 4, 5, 6]
