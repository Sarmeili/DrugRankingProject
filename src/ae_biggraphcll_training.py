from datahandler.cll_graph_handler import CllGraphHandler
import pickle

dh = CllGraphHandler()
cll_graph = dh.get_graph()
with open('graph_list.pkl', 'wb') as f:
    pickle.dump(cll_graph, f)