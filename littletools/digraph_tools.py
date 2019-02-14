import networkx as nx


def transitive_reduction(G):
    """ Returns transitive reduction of a directed graph

    The transitive reduction of G = (V,E) is a graph G- = (V,E-) such that
    for all v,w in V there is an edge (v,w) in E- if and only if (v,w) is
    in E and there is no path from v to w in G with length greater than 1.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    Returns
    -------
    NetworkX DiGraph
        The transitive reduction of `G`

    Raises
    ------
    NetworkXError
        If `G` is not a directed acyclic graph (DAG) transitive reduction is
        not uniquely defined and a :exc:`NetworkXError` exception is raised.

    References
    ----------
    https://en.wikipedia.org/wiki/Transitive_reduction

    """
    if not nx.algorithms.dag.is_directed_acyclic_graph(G):
        raise nx.NetworkXError(
            "Transitive reduction only uniquely defined on directed acyclic graphs.")
    TR = nx.DiGraph()
    TR.add_nodes_from(G.nodes())
    nx.set_node_attributes(TR, {n: d for n, d in G.nodes.items()})
    for u in G:
        u_edges = set(G[u])
        for v in G[u]:
            u_edges -= {y for x, y in nx.dfs_edges(G, v)}
        TR.add_edges_from((u, v) for v in u_edges)
    return TR


def find_roots(G):
    dfs_tree = nx.dfs_tree (G, depth_limit=0)
    return set(list([n1  for n1, n2 in dfs_tree.edges] ) + list(nx.isolates(G)))


def source_sink_generator (DiG):
    sink_nodes = [node for node, outdegree in DiG.out_degree(DiG.nodes()) if outdegree == 0]
    source_nodes = [node for node, indegree in DiG.in_degree(DiG.nodes()) if indegree == 0]
    for sink in sink_nodes:
        for source in source_nodes:
            for path in nx.all_simple_paths(DiG, source=source, target=sink):
                yield(path)