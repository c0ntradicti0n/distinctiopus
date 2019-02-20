import networkx as nx

from littletools.nested_list_tools import pairwise


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





def rs2graph(rs, G=None):
    """ Materialze the records of pyneo in networkx

        :param rs: records from query
        :return: nx.MultiDiGraph

    """
    # http://www.solasistim.net/posts/neo4j_to_networkx/
    if G == None:
        G = nx.MultiDiGraph()

    def add_nodes_and_edge(tup):
        (n1, n2, r) = tup

        G.add_node(n1.identity, kind=list(n1.labels), **n1)
        G.add_node(n2.identity, kind=list(n1.labels), **n2)
        G.add_edge(n1.identity, n2.identity, kind=r)

    result_tups = list(tuple(r) for r in rs)
    for tup in result_tups:
        add_nodes_and_edge(tup)
    return G


def neo4j2nx_mycel (pyneo4j, subgraph_marker):
    ''' Write a subgraph in neo4j, where a certain attribute is set, to nx

        :param pyneo4j: pyneo instance to call `run`
        :param subgraph_marker: the marker, that is set in the graph, just the property name
        :return: nx.MultiDiGraph

    '''
    query = """
    MATCH(a)-[r]->(b)
    WHERE EXISTS(a.{marker}) AND EXISTS(r.{marker}) AND EXISTS(r.{marker})
    RETURN a, b, r
    """.format (marker=subgraph_marker)
    record =  pyneo4j.run (query)
    G = rs2graph(record)
    return G


def neo4j2nx_root (pyneo4j, markers):
    ''' Write a subgraph in neo4j, where a certain attribute is set, to nx

        :param pyneo4j: pyneo instance to call `run`
        :param subgraph_marker: the marker, that is set in the graph, just the property name
        :return: nx.MultiDiGraph

    '''
    G = nx.MultiDiGraph()
    for head, child in pairwise(markers):
        query = r"""
        MATCH(a:{head})-[r]->(b:{child})
        RETURN a, b, type(r)
        """.format(head=head, child=child)
        record =  pyneo4j.run (query)
        G = rs2graph(record, G)
    return G

