import networkx as nx
from matplotlib import pyplot as plt
from sklearn.pipeline import FeatureUnion


def plot_pipeline(pipeline, save_fig_location):
    node_id = 2
    graph = [(1,2)]
    graph_labels = {1: "INPUT"}
    split_conserve = node_id
    branch_ends = []
    graph_color = ["lightgreen"]

    for item in pipeline.steps:
        if type(item[1])==FeatureUnion:

            for subitem in item[1].transformer_list:
                branch_start = True
                branch_len = len(subitem[1].steps)
                branch_step = 1
                
                for subsubitem in subitem[1].steps:
                    
                    if type(subsubitem[1])==FeatureUnion:
                        
                        for subsubsubitem in subsubitem[1].transformer_list:
                            branch_start_02 = True
                            branch_len_02 = len(subsubsubitem[1].steps)
                            branch_step_02 = 1
                    
                            if branch_start_02:
                                from_id = split_conserve_02
                                branch_start_02 = False
                            else:
                                from_id = node_id-1

                            graph_labels[node_id] = subsubsubitem[0]
                            graph.append((from_id, node_id))
                            graph_color.append('lightgray')

                            if branch_step_02 == branch_len_02:
                                branch_ends.append(node_id)

                            node_id += 1
                            branch_step_02 += 1
                    
                    else:
                        
                        if branch_start:
                            from_id = split_conserve
                            branch_start = False
                        else:
                            from_id = node_id-1

                        graph_labels[node_id] = subsubitem[0]
                        graph.append((from_id, node_id))
                        graph_color.append('lightgray')

                        if branch_step == branch_len:
                            branch_ends.append(node_id)
                        split_conserve_02 = node_id
                    node_id += 1
                    branch_step += 1
        else:
            
            if len(branch_ends)>0:

                for end in branch_ends:
                    graph.append((end, node_id))
                branch_ends=[]
                graph.append((node_id, node_id+1))
            else:
                graph.append((node_id, node_id+1))
            
            graph_labels[node_id] = item[0]
            graph_color.append('lightgray')
            split_conserve = node_id
        node_id +=1

    if len(branch_ends)>0:
        for end in branch_ends:
            graph.append((end, node_id-1))
        graph_color.append('orange')
        graph_labels[node_id-1] = "OUTPUT"
    else:
        graph_labels[node_id] = "OUTPUT"
        graph_color.append('orange')
        graph.append((node_id-1, node_id))

    G=nx.DiGraph()
    G.add_edges_from(graph)
    G.nodes(data=True)
    pos = nx.shell_layout(G)
    plt.figure(figsize=(15,10))
    nx.draw(G, pos, labels=graph_labels, node_size=2000, node_color=graph_color)
    plt.savefig(save_fig_location)
    plt.show()