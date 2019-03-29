import networkx as nx

from littletools.generator_tools import count_up
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
        (n1, n1_labels, n1_id, n2, n2_labels, n2_id, r) = tup.values()

        G.add_node(n1_id, kind=list(n1_labels), **n1)
        G.add_node(n2_id, kind=list(n2_labels), **n2)
        G.add_edge(n1_id, n2_id, kind=r)

    result_tups = list(tuple(r) for r in rs)
    for tup in rs:
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
        :param markers: list of string or tuples of strings. Tuples say, that different kind of nodes have to be
            collected at the same layer of the DiGraph
        :return: nx.MultiDiGraph

    '''
    def join_or_str(x, var):
        if isinstance(x, tuple):
            return "(" + var + ":"+(" or "+ var+":").join(x) + ")"
        else:
            return var+":"+x

    count =  count_up()

    G = nx.MultiDiGraph()

    rank = 0
    seen = []
    for head, child in pairwise(markers):
        if head==child:
            arrowed = '>'
            rank = next(count)
            r_diff = 0
        else:
            arrowed = ''
            rank = next(count)
            r_diff=1
        if not seen:
            in_nbunch = ''
        else:
            in_nbunch = 'and ID(a) in %s ' % str(seen)

        query = r"""
        MATCH(a)-[r]-{arrowed}(b)
        WHERE {label_head} and {label_child} {in_nbunch}
        SET a.rank={rank}, b.rank={rank}+{r_diff}
        RETURN a, labels(a), ID(a), b, labels(b), ID(b), type(r)
        """.format(label_head=join_or_str(head, 'a'), label_child=join_or_str(child, 'b'), rank=rank, r_diff=r_diff, arrowed=arrowed, in_nbunch=in_nbunch)
        record =  pyneo4j.run (query)
        seen += [r['ID(b)'] for r in record]

        G = rs2graph(record, G)
    return G

