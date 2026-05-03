from dsl.dataflow_dsl import Node, run_flow_graph


def increment(x): return x + 1
def square(x): return x * x


add_node = Node("add1", increment)
sq_node = Node("square", square)
add_node.then(sq_node)

data = list(range(10))
print(run_flow_graph(data, add_node))
