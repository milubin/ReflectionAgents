from dsl import ParallelDSL, Node, run_flow_graph


def test_map():
    dsl = ParallelDSL(backend="cpu")
    assert dsl.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


if __name__ == "__main__":
    test_map()
    print("All DSL tests passed!")
