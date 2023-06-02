from collections import defaultdict


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


# minimum cost so far for node (from start)
past_cost = defaultdict(lambda: 99)

heuristic_cost_to_go = {}
heuristic_cost_to_go[1] = 20
for i in range(2, 6):
    heuristic_cost_to_go[i] = 10
heuristic_cost_to_go[6] = 0

parent = {}
#  f f-Wert wird als nächster untersucht.
#
# f(x)=g(x)+h(x)
# g(x): bisherigen Kosten vom Startknoten aus, um x zu erreichen
# h(x): geschätzten Kosten von x bis zum Zielknoten
# c(a, b): Kosten der Kante


def a_star():
    open_list.insert(1, 20)
    past_cost[1] = 0
    print_all()
    while not open_list.empty():
        current = open_list.get_first()
        if current == end:
            return end
        closed_list.add(current)
        for nbr in get_neighbors(current):
            if nbr in closed_list:
                continue
            tentative_past_cost = past_cost[current] + cost[current][nbr]
            if tentative_past_cost < past_cost[nbr]:
                past_cost[nbr] = tentative_past_cost
                parent[nbr] = current
                est_total_cost = past_cost[nbr] + heuristic_cost_to_go[nbr]
                open_list.insert(nbr, est_total_cost)
            print_all()
    return None


def print_all():
    print_past_cost()
    print_optimist_cost_to_go()
    print_est_tot_cost()
    print_parent_nodes()
    print_open()
    print_closed()
    print("\n" + "="*40)


def print_path(item):
    path = []
    nxt = item
    while nxt != start:
        path = [nxt] + path
        nxt = parent[nxt]
    path = [start] + path
    print(path)


def print_past_cost():
    print()
    print(f"{'Past cost':<15}", end="")

    for i in range(1, 7):
        val = past_cost[i]
        print(f"{val:>3} | ", end="")


def print_optimist_cost_to_go():
    print()
    print(f"{'optimist ctg:':<15}", end="")

    for i in range(1, 7):
        val = heuristic_cost_to_go[i]
        print(f'{val:>3} | ', end="")


def print_est_tot_cost():
    print()
    print(f"{'est tot cost':<15}", end="")

    for i in range(1, 7):
        print(f'{open_list.get_est_total_cost(i):>3} | ', end="")

def print_parent_nodes():
    print()
    print(f"{'parent node:':<15}", end="")

    for i in range(1, 7):
        p = parent.get(i, "-")
        print(f'{p:>3} | ', end="")


def print_open():
    print("\nOPEN")
    lst = list(open_list.dict.items())
    lst.sort(key=lambda x: x[1])
    for l in lst:
        print(f"{l[0]} ({l[1]}), ", end="")
    print()

def print_closed():
    print("CLOSED")
    lst = list(closed_list)
    lst.sort()
    for l in lst:
        print(f"{l}, ", end="")
    print()


def test_cost():
    assert not cost[1][6]
    assert not cost[4][2]
    assert cost[4][1] == 12
    assert cost[3][6] == 15
    assert cost[3][2] == 27


def test_a_star():
    parent = a_star()
    print_path(parent)
