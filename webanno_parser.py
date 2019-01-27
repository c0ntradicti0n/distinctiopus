import pandas as pd
import numpy as np
import regex as re
import networkx as nx
from networkx.algorithms.components.connected import connected_components
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib
import pylab as plt
import datetime
import textwrap
from io import StringIO
from collections import namedtuple
import time
import logging

import sys

print(sys.version)


# anaphoras, ellipsis(either-or), comparison (more-than, most-else), exclusion (except-alone) are all things, that import information of the text


def to_graph(l):
    G = nx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current


def flatten(l):
    return [item for sublist in l for item in sublist]


def split_expand(df, lst_col, delim, debug=False):
    x = df.assign(**{lst_col: df[lst_col].str.split(delim)})
    return pd.DataFrame({
        col: np.repeat(x[col].values, x[lst_col].str.len())
        for col in x.columns.difference([lst_col])
    }).assign(**{lst_col: np.concatenate(x[lst_col].values)})[x.columns.tolist()]


class Webanno_Parser:
    string_nan = ":)"

    def __init__(self,
                 path,
                 dist_parts_col='Distinctionparts_is',
                 dist_sipas_Link_col='Distinctionparts_ROLE_webanno.custom.Distinctionparts:for_webanno.custom.DistinctionpartsForLink',
                 dist_level_BT_col='Distinctionlevel_BT_webanno.custom.Distinctionparts',
                 debug=False):
        self.csv_path = self.tsv2csv(path)
        self.df = self.load_df(self.csv_path)
        if debug:
            print(self.header)
            print(self.df)
        self.generalrels = list(self.extract_distpart_relations(self.df, dist_parts_col, debug=debug))
        self.distlevels = self.extract_distlevel_relations(self.df, dist_level_BT_col, debug=debug)

        try:
            self.distparts = [r for r in self.generalrels if r.pred == dist_sipas_Link_col][0]
        except IndexError:
            print("\nProbably wrong column name for Distinction Parts\n")
            print(self.header)
            print(list(self.generalrels))

            raise
        self.graph_info = self.hierarchy_graph_df(self.df, self.distparts, self.distlevels,
                                                  dist_parts_col, dist_sipas_Link_col, dist_level_BT_col, debug=debug)

        self.df[['sentence_id', 'word_id']] = self.df['swid'].str.split('-', expand=True)
        self.df[['sentence_id', 'word_id']] = self.df[['sentence_id', 'word_id']].astype(np.int64)
        return None

    def getdf(self):
        return self.df

    def tsv2csv(self, tsv_file, debug=False):
        """
        translate the tsv to a common csv-file with tab-separation,
        causes side_effects:
            list self.header
            list self.pred_arg_header
        are set

        caveto:
        the feature names in Webanno shouldn't end with "Link"
        """
        csv_file = tsv_file + ".csv"
        header = []
        pred_arg_header = []
        header_read = False
        header_line = re.compile(r"(?<=custom\.)(\w+)(?:\|([\w\.\:^\|]+))+")
        with open(tsv_file) as infile:
            with open(csv_file, 'w') as outfile:
                for s in infile:
                    if not s.strip():
                        continue
                    if s[0] == "#":
                        if not header_read:
                            m = header_line.search(s)
                            if m:
                                heads = m.group().split('|')
                                if debug:
                                    print(heads)

                                def realistic_column_namespan_beginning(h):
                                    return h.rfind('.') + 1

                                new = [heads[0] + '_' + h for h in heads[1:]]
                                pred_arg_header += [(
                                    heads[0] + '_' + h,
                                    heads[0] + '_' + heads[i + 1]
                                ) for i, h in enumerate(heads[0:-1])
                                    if (h.endswith("Link")
                                        and i > 0)]
                                header += new
                        continue
                    if header_read == False:
                        outfile.write("\t".join(['swid', 'span', 'token'] + header) + '\n')
                    outfile.write("\t" + s)
                    header_read = True
        self.header = ['swid', 'span', 'token'] + header
        if debug: print("self.header", self.header)
        self.pred_arg_header = pred_arg_header
        if debug: print("pred_arg_header", pred_arg_header)
        return csv_file

    def load_df(self, csv_file, debug=False):
        df = pd.read_csv(csv_file, header=0, index_col=False, sep='\t+', na_values="_", engine='python')
        if debug: print("READ CSV")
        if debug: print(df.columns)
        if debug: print(df)
        df = df.dropna(axis='columns', how='all')
        df = df.fillna(self.string_nan).astype(str)
        return df

    DistpartRelation = namedtuple('DistpartRelation', ['pred', 'arg', 'edges', 'labels1', 'labels2'])

    def extract_distpart_relations(self, df, dist_parts_col, debug=False):
        if debug: print(self.pred_arg_header)
        for pred, arg in self.pred_arg_header:

            a = df[pred].str.extractall("(?P<pointing_name>\w+)>(?P<to_name>\w+)\[(?P<pointing_addr>\d+)\]")
            b = df[arg].str.extractall("(?<!_)\[(?P<to_addr>\d+)\]")

            try:
                res = a.merge(b, how='outer', left_index=True, right_index=True).drop_duplicates()
            except ValueError:
                if debug:
                    print("Could not find relations for \n    " + pred + "\nand\n    " + arg)
                continue

            if debug:
                print("RESULT")
                print(res)

            res = res.dropna()

            all_relation_edges = list(zip(res['pointing_addr'], res['to_addr']))

            node_label_tuples = list(zip(res['pointing_addr'], res['pointing_name'])) + list(
                zip(res['to_addr'], res['to_name']))
            node_labels_dict1 = {i: {'label': kind} for i, kind in
                                 node_label_tuples}  # as nested dict {node: {attr: label}}
            node_labels_dict2 = {i: kind for i, kind in node_label_tuples}  # as simple dict {node: label}
            yield self.DistpartRelation(pred, arg, all_relation_edges, node_labels_dict1, node_labels_dict2)

    DistlevelRelation = namedtuple('DistlevelRelation', ['vertex', 'chain', 'labels1', 'labels2'])

    def extract_distlevel_relations(self, df, dist_level_col, debug=False):
        filtered_df = df[df[dist_level_col] != self.string_nan]
        extracted_df = filtered_df[dist_level_col] \
            .str.extractall(".+\[(?P<this_token>\d+)\_(?P<other_token>\d+)\]") \
            .dropna()
        if debug:
            print(extracted_df)
        pair_distinctions = list(zip(extracted_df['this_token'], extracted_df['other_token']))
        graph = to_graph(pair_distinctions)
        grouped_distinctions = list([list(e) for e in connected_components(graph)])
        if debug:
            print(grouped_distinctions)

        def level_node_name(i, l):
            return 'Distinction ' + str(i)  # + "~".join(l)

        distinction_level_edges = flatten(
            [[(str(n), level_node_name(i, group)) for n in group] for i, group in enumerate(grouped_distinctions)])
        distlevel_labels_dict1 = {level_node_name(i, group): {'label': 'level'} for i, group in
                                  enumerate(grouped_distinctions)}
        distlevel_labels_dict2 = {level_node_name(i, group): 'level' for i, group in enumerate(grouped_distinctions)}

        if debug:
            print("RELATION EDGES parts_of_distinctions", distinction_level_edges)
            print("PAIR DISTINCTIONS", pair_distinctions)
            print("RELATION NODE LABELS", distlevel_labels_dict1)
            print("RELATION LABEL DICT", distlevel_labels_dict2)
        return self.DistlevelRelation(distinction_level_edges, pair_distinctions, distlevel_labels_dict1,
                                      distlevel_labels_dict2)

    GraphEvaluation = namedtuple('GraphEvaluation', [
        'whole_graph',
        'sub_levels_graph',
        'sub_parts_graph',
        'labels1',
        'labels2',
        'connected_distparts',
        'connected_distlevels',
        'text_dict',
        'kind_dict',
        'grouped_df',
        'merged_df',
        'graph_df'])

    def hierarchy_graph_df(self, df, distparts, distlevels, dist_parts_col, dist_sipas_Link_col, dist_level_col,
                           debug=False):
        if debug:
            print(distparts)
            print("\n\n\n")
            print(distlevels)

        def append_dict(d1, d2):
            return dict(d1, **d2);
            d4.update(d3)

        labels1 = append_dict(distparts.labels1, distlevels.labels1)
        labels2 = append_dict(distparts.labels2, distlevels.labels2)
        all_edges = distparts.edges + distlevels.vertex

        whole_graph = to_graph(all_edges)
        sub_levels_graph = to_graph(distlevels.chain)
        sub_parts_graph = to_graph(distparts.edges)

        nx.set_node_attributes(whole_graph, labels1)

        connected_distparts = list([list(e) for e in connected_components(sub_parts_graph)])
        connected_distlevels = list([list(e) for e in connected_components(sub_levels_graph)])

        # find the text for the nodes in the dataframe
        # step 1: first df with only the layers enumerated and a list of the nodes that belong to that
        graph_df = pd.DataFrame({'sidesoflevels': connected_distlevels, 'layer': range(0, len(connected_distlevels))})
        if debug:
            print(graph_df)

        # step 2: expand these lists and that we call that the levels,
        #         with these levels nodes are meant that belong to the different connected sides of the distinctionparts
        graph_df = pd.DataFrame(
            [(d, tup.layer, tup.sidesoflevels) for tup in graph_df.itertuples() for d in tup.sidesoflevels])
        graph_df = graph_df.rename(columns={0: 'level', 1: 'layer', 2: 'sidesoflevels'})

        if debug:
            print(graph_df)

        # step 3: to this we find the lists of nodes of the connected sides of the distinctionparts
        def sublist_if_intersection(x):
            return flatten([l for l in connected_distparts if x in l])

        # step 4: we expand again these new lists of connected components
        graph_df['partsofsides'] = graph_df['level'].apply(sublist_if_intersection)
        if debug:
            print(graph_df)

        graph_df = pd.DataFrame([(d, tup.level, tup.layer, tup.sidesoflevels, tup.partsofsides)
                                 for tup in graph_df.itertuples() for d in tup.partsofsides])
        if debug:
            print(graph_df)
        graph_df = graph_df.rename(columns={0: 'part', 1: 'level', 2: 'layer', 3: 'levels', 4: 'distinction_parts'})
        if debug:
            print(graph_df)

        graph_df['this_distinction_part'] = graph_df['part'].apply(
            lambda x: "".join([labels2[x], '[', x, ']']))  # for merging
        graph_df['kind'] = graph_df['part'].apply(lambda x: labels2[x])  # for colouring

        df = split_expand(df, dist_parts_col, '|')

        res = graph_df.merge(df, right_on=dist_parts_col, left_on='this_distinction_part', how='outer').fillna(
            self.string_nan)

        if debug:
            print(res)
        interesting_cols = ['layer', 'level', 'part']

        grouped_token = res.groupby(interesting_cols)['token'].apply(lambda x: "%s" % ' '.join(x))
        grouped_token = grouped_token.drop(self.string_nan, level=0)
        grouped_token = grouped_token.reset_index()

        text_dict = dict(zip(grouped_token['part'], grouped_token['token']))

        for node, text in {n: t for n, t in text_dict.items() if t == self.string_nan}.items():
            mask = df[dist_parts_col].str.contains('[' + node + ']', regex=False)
            if mask.any():
                if debug:
                    print('[' + node + ']')
                text_dict[node] = " ".join(list(df[mask]['token']))

        grouped_kind = res.groupby(interesting_cols)['kind'].apply(lambda x: "%s" % str(x.unique()[0]))
        grouped_kind = grouped_kind.drop(self.string_nan, level=0)
        grouped_kind = grouped_kind.reset_index()

        kind_dict = dict(zip(grouped_kind['part'], grouped_kind['kind']))

        if debug:
            print(text_dict)

            with pd.option_context('display.max_rows', None, 'display.max_columns', 3, "display.latex.multirow", True):
                print(grouped_token)

        return self.GraphEvaluation(
            whole_graph,
            sub_levels_graph,
            sub_parts_graph,
            labels1,
            labels2,
            connected_distparts,
            connected_distlevels,
            text_dict,
            kind_dict,
            grouped_token,
            res,
            graph_df)

    def paint_graphviz(self, path, debug=False):
        graph = self.graph_info.whole_graph
        text_dict = {key: textwrap.fill(text, width=35)
                     for key, text in self.graph_info.text_dict.items()
                     if key in graph.nodes}
        if debug:
            print({n: t for n, t in text_dict.items() if t == self.string_nan}.items())
        df = self.df
        for node, text in {n: t for n, t in text_dict.items() if t == self.string_nan}.items():
            if debug:
                print(node)
            for row in range(df.shape[0]):  # df is the DataFrame
                for col in range(df.shape[1]):
                    try:
                        if debug:
                            print(df.get_value(row, col))
                            print
                    except:
                        continue
                    if (('[' + node + ']' in str(df.get_value(row, col)))):
                        if debug:
                            print(row, col)
                            break

        kind_dict = self.graph_info.kind_dict

        if debug:
            print(kind_dict)
        color_dict = {
            'entity': 'cornflowerblue',
            'predicate': 'darkorange',
            'modificator': 'forestgreen',
            'example': 'olive',
            'marker': 'teal',
            'what?': 'purple'
        }
        color_map = []
        for node in graph:
            if not node in kind_dict:
                if 'Distinction' in str(node):
                    color_map.append('black')
                else:
                    print('node ' + str(node) + ' not classified as distinction part')
                    color_map.append('magenta')
                continue
            if kind_dict[node] in color_dict:
                color_map.append(color_dict[kind_dict[node]])
            else:
                color_map.append('yellow')

        plt.figure(4, figsize=(35, 23))
        nodeFontSize = 6
        nodeSize = 300
        nodeColorList = color_map
        # edgeColorList   = getEdgeColor(graph.edges())

        # Graphiz tunning
        prog = 'dot'
        args = ' -Gdpi=250 -Gnodesep=3 -Granksep=6 -Gpad=1.9 -Grankdir=TD '
        root = None
        pos = graphviz_layout(graph, prog='neato', root=root, args=args)

        nx.draw(graph,
                pos=pos,
                with_labels=True,
                node_color=nodeColorList,
                # edge_color  = edgeColorList,
                font_size=nodeFontSize,
                node_size=nodeSize, )

        for p in pos:
            yOffSet = -10
            xOffSet = 10
            pos[p] = (pos[p][0] + xOffSet, pos[p][1] + yOffSet)

        labelDescr = nx.draw_networkx_labels(
            graph,
            pos=pos,
            # labels     = label_dict,
            labels=text_dict,
            font_size=nodeFontSize, )

        # for n,t in labelDescr.items():
        #    finDegree = 70
        #    t.set_rotation(finDegree)

        now = datetime.datetime.now()
        plt.savefig(path)

        if debug:
            print(self.graph_info.connected_distparts)
            print(self.graph_info.connected_distlevels)
            print(list([list(e) for e in connected_components(graph)]))

        return None


def main():
    img_folder = './printedphraphs/'
    now = datetime.datetime.now()

    webannoparsed = Webanno_Parser("corpus/aristotle-categories-edghill-spell-checked.tsv")
    webannoparsed.paint_graphviz(img_folder + "graphviz " + '_graphviz.png')

    df = webannoparsed.getdf()

    print(df[['token', 'swid', 'sentence_id', 'word_id']])
    # """+ now.strftime("%Y-%m-%d %H:%M:%S")"""


if __name__ == '__main__':
    main()