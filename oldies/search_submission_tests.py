# coding=utf-8
import pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from submission import PriorityQueue, a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, uniform_cost_search, haversine_dist_heuristic, \
    tridirectional_upgraded, custom_heuristic
from visualize_graph import plot_search


class TestPriorityQueue(unittest.TestCase):
    """Test Priority Queue implementation"""

    
        
    
    def test_append_and_pop(self):
        """Test the append and pop functions"""
        queue = PriorityQueue()
        temp_list = []

        for _ in range(10):
            a = random.randint(0, 10000)
            queue.append((a, 'a'))
            temp_list.append(a)

        temp_list = sorted(temp_list)

        for item in temp_list:
            popped = queue.pop()
            self.assertEqual(popped[0], item)

    def test_fifo_property(self):
        "Test the fifo property for nodes with same priority"
        queue = PriorityQueue()
        temp_list = [(1,'b'), (1, 'c'), (1, 'a')]

        for node in temp_list:
            queue.append(node)
        
        for expected_node in temp_list:
            actual_node = queue.pop()
            self.assertEqual(actual_node[-1], expected_node[-1])

class TestBasicSearch(unittest.TestCase):
    """Test the simple search algorithms: BFS, UCS, A*"""

    
    def setUp(self):
        """Romania map data from Russell and Norvig, Chapter 3."""
        with open('romania_graph.pickle', 'rb') as rom:
            romania = pickle.load(rom)
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()
        
        
        

    def test_bfs(self):
        """Test and visualize breadth-first search"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}

        self.romania.reset_search()
        path = breadth_first_search(self.romania, start, goal)

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path,
                        title='test_bfs blue=start, yellow=goal, green=explored')

    def test_bfs_num_explored(self):
        """Test BFS for correct path and number of explored nodes"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}

        self.romania.reset_search()
        path = breadth_first_search(self.romania, start, goal)

        self.assertEqual(['a', 's', 'f', 'b', 'u'], path)   # Check for correct path

        explored_nodes = sum(list(self.romania.explored_nodes().values()))
        self.assertLessEqual(explored_nodes, 10)    # Compare explored nodes to reference implementation

    def test_bfs_empty_path(self):
        start = "a"
        goal = "a"
        path = breadth_first_search(self.romania, start, goal)
        self.assertEqual(path, [])

    def test_ucs(self):
        """TTest and visualize uniform-cost search"""
        start = 'a'
        goal = 'u'
        
        # correct_tests_ucs = {}
        
        # nodes = ['a','z','o','s','t','l','m','d','c','r','p','b','f','g','u','h','e','v','i','n']
        # for start in nodes:
            
        #     for goal in nodes:
        

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}
    
        self.romania.reset_search()
        path = uniform_cost_search(self.romania, start, goal)
        
        # correct_tests_ucs[(start,goal)] = path
        
        
        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path,
                        title='test_ucs blue=start, yellow=goal, green=explored')
        # print(correct_tests_ucs)

    def test_ucs_num_explored(self):
        """Test UCS for correct path and number of explored nodes"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}

        self.romania.reset_search()
        path = uniform_cost_search(self.romania, start, goal)

        self.assertEqual(path, ['a', 's', 'r', 'p', 'b', 'u'])   # Check for correct path

        explored_nodes = sum(list(self.romania.explored_nodes().values()))
        self.assertEqual(explored_nodes, 13)    # Compare explored nodes to reference implementation
        
    
    

    def test_a_star(self):
        """Test and visualize A* search"""
        start = 'a'
        goal = 'u'
        
        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}
        
        self.romania.reset_search()
        path = a_star(self.romania, start, goal)

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path,
                        title='test_astar blue=start, yellow=goal, green=explored')

    
    def test_a_star_num_explored(self):
        """Test A* for correct path and number of explored nodes"""
        start = 'a'
        goal = 'u'

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}

        self.romania.reset_search()
        path = a_star(self.romania, start, goal)

        self.assertEqual(path, ['a', 's', 'r', 'p', 'b', 'u'])   # Check for correct path

        explored_nodes = sum(list(self.romania.explored_nodes().values()))
        self.assertEqual(explored_nodes, 8)    # Compare explored nodes to reference implementation

    @staticmethod
    def draw_graph(graph, node_positions=None, start=None, goal=None,
                   path=None, title=''):
        """Visualize results of graph search"""
        explored = [key for key in graph.explored_nodes() if graph.explored_nodes()[key] > 0]

        labels = {}
        for node in graph:
            labels[node] = node

        if node_positions is None:
            node_positions = networkx.spring_layout(graph)

        networkx.draw_networkx_nodes(graph, node_positions)
        networkx.draw_networkx_edges(graph, node_positions, style='dashed')
        networkx.draw_networkx_labels(graph, node_positions, labels)

        networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
                                     node_color='g')
        edge_labels = networkx.get_edge_attributes(graph, 'weight')
        networkx.draw_networkx_edge_labels(graph, node_positions, edge_labels=edge_labels)
        
        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
            networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
                                         edge_color='b')

        if start:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[start], node_color='b')

        if goal:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[goal], node_color='y')

        plt.title(title)
        plt.plot()
        plt.show()


class TestBidirectionalSearch(unittest.TestCase):
    """Test the bidirectional search algorithms: UCS, A*"""

    def setUp(self):
        """Load Atlanta map data"""
        with open('atlanta_osm.pickle', 'rb') as atl:
            atlanta = pickle.load(atl)
        self.atlanta = ExplorableGraph(atlanta)
        self.atlanta.reset_search()

        with open('romania_graph.pickle', 'rb') as rom:
            romania = pickle.load(rom)
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()

    def test_bidirectional_ucs(self):
        """Test and generate GeoJSON for bidirectional UCS search"""
        path = bidirectional_ucs(self.atlanta, '69581003', '69581000')
        all_explored = self.atlanta.explored_nodes()
        plot_search(self.atlanta, 'atlanta_search_bidir_ucs.json', path,
                    all_explored)
    
    def test_bidirectional_ucs_romania(self):
        """Test Bi-UCS and visualize"""
        correct_tests_ucs = {('a', 'a'): [], ('a', 'z'): ['a', 'z'], ('a', 'o'): ['a', 'z', 'o'], ('a', 's'): ['a', 's'], ('a', 't'): ['a', 't'], ('a', 'l'): ['a', 't', 'l'], ('a', 'm'): ['a', 't', 'l', 'm'], ('a', 'd'): ['a', 't', 'l', 'm', 'd'], ('a', 'c'): ['a', 's', 'r', 'c'], ('a', 'r'): ['a', 's', 'r'], ('a', 'p'): ['a', 's', 'r', 'p'], ('a', 'b'): ['a', 's', 'r', 'p', 'b'], ('a', 'f'): ['a', 's', 'f'], ('a', 'g'): ['a', 's', 'r', 'p', 'b', 'g'], ('a', 'u'): ['a', 's', 'r', 'p', 'b', 'u'], ('a', 'h'): ['a', 's', 'r', 'p', 'b', 'u', 'h'], ('a', 'e'): ['a', 's', 'r', 'p', 'b', 'u', 'h', 'e'], ('a', 'v'): ['a', 's', 'r', 'p', 'b', 'u', 'v'], ('a', 'i'): ['a', 's', 'r', 'p', 'b', 'u', 'v', 'i'], ('a', 'n'): ['a', 's', 'r', 'p', 'b', 'u', 'v', 'i', 'n'], ('z', 'a'): ['z', 'a'], ('z', 'z'): [], ('z', 'o'): ['z', 'o'], ('z', 's'): ['z', 'a', 's'], ('z', 't'): ['z', 'a', 't'], ('z', 'l'): ['z', 'a', 't', 'l'], ('z', 'm'): ['z', 'a', 't', 'l', 'm'], ('z', 'd'): ['z', 'a', 't', 'l', 'm', 'd'], ('z', 'c'): ['z', 'a', 's', 'r', 'c'], ('z', 'r'): ['z', 'a', 's', 'r'], ('z', 'p'): ['z', 'a', 's', 'r', 'p'], ('z', 'b'): ['z', 'a', 's', 'r', 'p', 'b'], ('z', 'f'): ['z', 'a', 's', 'f'], ('z', 'g'): ['z', 'a', 's', 'r', 'p', 'b', 'g'], ('z', 'u'): ['z', 'a', 's', 'r', 'p', 'b', 'u'], ('z', 'h'): ['z', 'a', 's', 'r', 'p', 'b', 'u', 'h'], ('z', 'e'): ['z', 'a', 's', 'r', 'p', 'b', 'u', 'h', 'e'], ('z', 'v'): ['z', 'a', 's', 'r', 'p', 'b', 'u', 'v'], ('z', 'i'): ['z', 'a', 's', 'r', 'p', 'b', 'u', 'v', 'i'], ('z', 'n'): ['z', 'a', 's', 'r', 'p', 'b', 'u', 'v', 'i', 'n'], ('o', 'a'): ['o', 'z', 'a'], ('o', 'z'): ['o', 'z'], ('o', 'o'): [], ('o', 's'): ['o', 's'], ('o', 't'): ['o', 'z', 'a', 't'], ('o', 'l'): ['o', 'z', 'a', 't', 'l'], ('o', 'm'): ['o', 'z', 'a', 't', 'l', 'm'], ('o', 'd'): ['o', 's', 'r', 'c', 'd'], ('o', 'c'): ['o', 's', 'r', 'c'], ('o', 'r'): ['o', 's', 'r'], ('o', 'p'): ['o', 's', 'r', 'p'], ('o', 'b'): ['o', 's', 'r', 'p', 'b'], ('o', 'f'): ['o', 's', 'f'], ('o', 'g'): ['o', 's', 'r', 'p', 'b', 'g'], ('o', 'u'): ['o', 's', 'r', 'p', 'b', 'u'], ('o', 'h'): ['o', 's', 'r', 'p', 'b', 'u', 'h'], ('o', 'e'): ['o', 's', 'r', 'p', 'b', 'u', 'h', 'e'], ('o', 'v'): ['o', 's', 'r', 'p', 'b', 'u', 'v'], ('o', 'i'): ['o', 's', 'r', 'p', 'b', 'u', 'v', 'i'], ('o', 'n'): ['o', 's', 'r', 'p', 'b', 'u', 'v', 'i', 'n'], ('s', 'a'): ['s', 'a'], ('s', 'z'): ['s', 'a', 'z'], ('s', 'o'): ['s', 'o'], ('s', 's'): [], ('s', 't'): ['s', 'a', 't'], ('s', 'l'): ['s', 'a', 't', 'l'], ('s', 'm'): ['s', 'r', 'c', 'd', 'm'], ('s', 'd'): ['s', 'r', 'c', 'd'], ('s', 'c'): ['s', 'r', 'c'], ('s', 'r'): ['s', 'r'], ('s', 'p'): ['s', 'r', 'p'], ('s', 'b'): ['s', 'r', 'p', 'b'], ('s', 'f'): ['s', 'f'], ('s', 'g'): ['s', 'r', 'p', 'b', 'g'], ('s', 'u'): ['s', 'r', 'p', 'b', 'u'], ('s', 'h'): ['s', 'r', 'p', 'b', 'u', 'h'], ('s', 'e'): ['s', 'r', 'p', 'b', 'u', 'h', 'e'], ('s', 'v'): ['s', 'r', 'p', 'b', 'u', 'v'], ('s', 'i'): ['s', 'r', 'p', 'b', 'u', 'v', 'i'], ('s', 'n'): ['s', 'r', 'p', 'b', 'u', 'v', 'i', 'n'], ('t', 'a'): ['t', 'a'], ('t', 'z'): ['t', 'a', 'z'], ('t', 'o'): ['t', 'a', 'z', 'o'], ('t', 's'): ['t', 'a', 's'], ('t', 't'): [], ('t', 'l'): ['t', 'l'], ('t', 'm'): ['t', 'l', 'm'], ('t', 'd'): ['t', 'l', 'm', 'd'], ('t', 'c'): ['t', 'l', 'm', 'd', 'c'], ('t', 'r'): ['t', 'a', 's', 'r'], ('t', 'p'): ['t', 'a', 's', 'r', 'p'], ('t', 'b'): ['t', 'a', 's', 'r', 'p', 'b'], ('t', 'f'): ['t', 'a', 's', 'f'], ('t', 'g'): ['t', 'a', 's', 'r', 'p', 'b', 'g'], ('t', 'u'): ['t', 'a', 's', 'r', 'p', 'b', 'u'], ('t', 'h'): ['t', 'a', 's', 'r', 'p', 'b', 'u', 'h'], ('t', 'e'): ['t', 'a', 's', 'r', 'p', 'b', 'u', 'h', 'e'], ('t', 'v'): ['t', 'a', 's', 'r', 'p', 'b', 'u', 'v'], ('t', 'i'): ['t', 'a', 's', 'r', 'p', 'b', 'u', 'v', 'i'], ('t', 'n'): ['t', 'a', 's', 'r', 'p', 'b', 'u', 'v', 'i', 'n'], ('l', 'a'): ['l', 't', 'a'], ('l', 'z'): ['l', 't', 'a', 'z'], ('l', 'o'): ['l', 't', 'a', 'z', 'o'], ('l', 's'): ['l', 't', 'a', 's'], ('l', 't'): ['l', 't'], ('l', 'l'): [], ('l', 'm'): ['l', 'm'], ('l', 'd'): ['l', 'm', 'd'], ('l', 'c'): ['l', 'm', 'd', 'c'], ('l', 'r'): ['l', 'm', 'd', 'c', 'r'], ('l', 'p'): ['l', 'm', 'd', 'c', 'p'], ('l', 'b'): ['l', 'm', 'd', 'c', 'p', 'b'], ('l', 'f'): ['l', 't', 'a', 's', 'f'], ('l', 'g'): ['l', 'm', 'd', 'c', 'p', 'b', 'g'], ('l', 'u'): ['l', 'm', 'd', 'c', 'p', 'b', 'u'], ('l', 'h'): ['l', 'm', 'd', 'c', 'p', 'b', 'u', 'h'], ('l', 'e'): ['l', 'm', 'd', 'c', 'p', 'b', 'u', 'h', 'e'], ('l', 'v'): ['l', 'm', 'd', 'c', 'p', 'b', 'u', 'v'], ('l', 'i'): ['l', 'm', 'd', 'c', 'p', 'b', 'u', 'v', 'i'], ('l', 'n'): ['l', 'm', 'd', 'c', 'p', 'b', 'u', 'v', 'i', 'n'], ('m', 'a'): ['m', 'l', 't', 'a'], ('m', 'z'): ['m', 'l', 't', 'a', 'z'], ('m', 'o'): ['m', 'l', 't', 'a', 'z', 'o'], ('m', 's'): ['m', 'd', 'c', 'r', 's'], ('m', 't'): ['m', 'l', 't'], ('m', 'l'): ['m', 'l'], ('m', 'm'): [], ('m', 'd'): ['m', 'd'], ('m', 'c'): ['m', 'd', 'c'], ('m', 'r'): ['m', 'd', 'c', 'r'], ('m', 'p'): ['m', 'd', 'c', 'p'], ('m', 'b'): ['m', 'd', 'c', 'p', 'b'], ('m', 'f'): ['m', 'd', 'c', 'r', 's', 'f'], ('m', 'g'): ['m', 'd', 'c', 'p', 'b', 'g'], ('m', 'u'): ['m', 'd', 'c', 'p', 'b', 'u'], ('m', 'h'): ['m', 'd', 'c', 'p', 'b', 'u', 'h'], ('m', 'e'): ['m', 'd', 'c', 'p', 'b', 'u', 'h', 'e'], ('m', 'v'): ['m', 'd', 'c', 'p', 'b', 'u', 'v'], ('m', 'i'): ['m', 'd', 'c', 'p', 'b', 'u', 'v', 'i'], ('m', 'n'): ['m', 'd', 'c', 'p', 'b', 'u', 'v', 'i', 'n'], ('d', 'a'): ['d', 'm', 'l', 't', 'a'], ('d', 'z'): ['d', 'm', 'l', 't', 'a', 'z'], ('d', 'o'): ['d', 'c', 'r', 's', 'o'], ('d', 's'): ['d', 'c', 'r', 's'], ('d', 't'): ['d', 'm', 'l', 't'], ('d', 'l'): ['d', 'm', 'l'], ('d', 'm'): ['d', 'm'], ('d', 'd'): [], ('d', 'c'): ['d', 'c'], ('d', 'r'): ['d', 'c', 'r'], ('d', 'p'): ['d', 'c', 'p'], ('d', 'b'): ['d', 'c', 'p', 'b'], ('d', 'f'): ['d', 'c', 'r', 's', 'f'], ('d', 'g'): ['d', 'c', 'p', 'b', 'g'], ('d', 'u'): ['d', 'c', 'p', 'b', 'u'], ('d', 'h'): ['d', 'c', 'p', 'b', 'u', 'h'], ('d', 'e'): ['d', 'c', 'p', 'b', 'u', 'h', 'e'], ('d', 'v'): ['d', 'c', 'p', 'b', 'u', 'v'], ('d', 'i'): ['d', 'c', 'p', 'b', 'u', 'v', 'i'], ('d', 'n'): ['d', 'c', 'p', 'b', 'u', 'v', 'i', 'n'], ('c', 'a'): ['c', 'r', 's', 'a'], ('c', 'z'): ['c', 'r', 's', 'a', 'z'], ('c', 'o'): ['c', 'r', 's', 'o'], ('c', 's'): ['c', 'r', 's'], ('c', 't'): ['c', 'd', 'm', 'l', 't'], ('c', 'l'): ['c', 'd', 'm', 'l'], ('c', 'm'): ['c', 'd', 'm'], ('c', 'd'): ['c', 'd'], ('c', 'c'): [], ('c', 'r'): ['c', 'r'], ('c', 'p'): ['c', 'p'], ('c', 'b'): ['c', 'p', 'b'], ('c', 'f'): ['c', 'r', 's', 'f'], ('c', 'g'): ['c', 'p', 'b', 'g'], ('c', 'u'): ['c', 'p', 'b', 'u'], ('c', 'h'): ['c', 'p', 'b', 'u', 'h'], ('c', 'e'): ['c', 'p', 'b', 'u', 'h', 'e'], ('c', 'v'): ['c', 'p', 'b', 'u', 'v'], ('c', 'i'): ['c', 'p', 'b', 'u', 'v', 'i'], ('c', 'n'): ['c', 'p', 'b', 'u', 'v', 'i', 'n'], ('r', 'a'): ['r', 's', 'a'], ('r', 'z'): ['r', 's', 'a', 'z'], ('r', 'o'): ['r', 's', 'o'], ('r', 's'): ['r', 's'], ('r', 't'): ['r', 's', 'a', 't'], ('r', 'l'): ['r', 'c', 'd', 'm', 'l'], ('r', 'm'): ['r', 'c', 'd', 'm'], ('r', 'd'): ['r', 'c', 'd'], ('r', 'c'): ['r', 'c'], ('r', 'r'): [], ('r', 'p'): ['r', 'p'], ('r', 'b'): ['r', 'p', 'b'], ('r', 'f'): ['r', 's', 'f'], ('r', 'g'): ['r', 'p', 'b', 'g'], ('r', 'u'): ['r', 'p', 'b', 'u'], ('r', 'h'): ['r', 'p', 'b', 'u', 'h'], ('r', 'e'): ['r', 'p', 'b', 'u', 'h', 'e'], ('r', 'v'): ['r', 'p', 'b', 'u', 'v'], ('r', 'i'): ['r', 'p', 'b', 'u', 'v', 'i'], ('r', 'n'): ['r', 'p', 'b', 'u', 'v', 'i', 'n'], ('p', 'a'): ['p', 'r', 's', 'a'], ('p', 'z'): ['p', 'r', 's', 'a', 'z'], ('p', 'o'): ['p', 'r', 's', 'o'], ('p', 's'): ['p', 'r', 's'], ('p', 't'): ['p', 'r', 's', 'a', 't'], ('p', 'l'): ['p', 'c', 'd', 'm', 'l'], ('p', 'm'): ['p', 'c', 'd', 'm'], ('p', 'd'): ['p', 'c', 'd'], ('p', 'c'): ['p', 'c'], ('p', 'r'): ['p', 'r'], ('p', 'p'): [], ('p', 'b'): ['p', 'b'], ('p', 'f'): ['p', 'r', 's', 'f'], ('p', 'g'): ['p', 'b', 'g'], ('p', 'u'): ['p', 'b', 'u'], ('p', 'h'): ['p', 'b', 'u', 'h'], ('p', 'e'): ['p', 'b', 'u', 'h', 'e'], ('p', 'v'): ['p', 'b', 'u', 'v'], ('p', 'i'): ['p', 'b', 'u', 'v', 'i'], ('p', 'n'): ['p', 'b', 'u', 'v', 'i', 'n'], ('b', 'a'): ['b', 'p', 'r', 's', 'a'], ('b', 'z'): ['b', 'p', 'r', 's', 'a', 'z'], ('b', 'o'): ['b', 'p', 'r', 's', 'o'], ('b', 's'): ['b', 'p', 'r', 's'], ('b', 't'): ['b', 'p', 'r', 's', 'a', 't'], ('b', 'l'): ['b', 'p', 'c', 'd', 'm', 'l'], ('b', 'm'): ['b', 'p', 'c', 'd', 'm'], ('b', 'd'): ['b', 'p', 'c', 'd'], ('b', 'c'): ['b', 'p', 'c'], ('b', 'r'): ['b', 'p', 'r'], ('b', 'p'): ['b', 'p'], ('b', 'b'): [], ('b', 'f'): ['b', 'f'], ('b', 'g'): ['b', 'g'], ('b', 'u'): ['b', 'u'], ('b', 'h'): ['b', 'u', 'h'], ('b', 'e'): ['b', 'u', 'h', 'e'], ('b', 'v'): ['b', 'u', 'v'], ('b', 'i'): ['b', 'u', 'v', 'i'], ('b', 'n'): ['b', 'u', 'v', 'i', 'n'], ('f', 'a'): ['f', 's', 'a'], ('f', 'z'): ['f', 's', 'a', 'z'], ('f', 'o'): ['f', 's', 'o'], ('f', 's'): ['f', 's'], ('f', 't'): ['f', 's', 'a', 't'], ('f', 'l'): ['f', 's', 'a', 't', 'l'], ('f', 'm'): ['f', 's', 'r', 'c', 'd', 'm'], ('f', 'd'): ['f', 's', 'r', 'c', 'd'], ('f', 'c'): ['f', 's', 'r', 'c'], ('f', 'r'): ['f', 's', 'r'], ('f', 'p'): ['f', 's', 'r', 'p'], ('f', 'b'): ['f', 'b'], ('f', 'f'): [], ('f', 'g'): ['f', 'b', 'g'], ('f', 'u'): ['f', 'b', 'u'], ('f', 'h'): ['f', 'b', 'u', 'h'], ('f', 'e'): ['f', 'b', 'u', 'h', 'e'], ('f', 'v'): ['f', 'b', 'u', 'v'], ('f', 'i'): ['f', 'b', 'u', 'v', 'i'], ('f', 'n'): ['f', 'b', 'u', 'v', 'i', 'n'], ('g', 'a'): ['g', 'b', 'p', 'r', 's', 'a'], ('g', 'z'): ['g', 'b', 'p', 'r', 's', 'a', 'z'], ('g', 'o'): ['g', 'b', 'p', 'r', 's', 'o'], ('g', 's'): ['g', 'b', 'p', 'r', 's'], ('g', 't'): ['g', 'b', 'p', 'r', 's', 'a', 't'], ('g', 'l'): ['g', 'b', 'p', 'c', 'd', 'm', 'l'], ('g', 'm'): ['g', 'b', 'p', 'c', 'd', 'm'], ('g', 'd'): ['g', 'b', 'p', 'c', 'd'], ('g', 'c'): ['g', 'b', 'p', 'c'], ('g', 'r'): ['g', 'b', 'p', 'r'], ('g', 'p'): ['g', 'b', 'p'], ('g', 'b'): ['g', 'b'], ('g', 'f'): ['g', 'b', 'f'], ('g', 'g'): [], ('g', 'u'): ['g', 'b', 'u'], ('g', 'h'): ['g', 'b', 'u', 'h'], ('g', 'e'): ['g', 'b', 'u', 'h', 'e'], ('g', 'v'): ['g', 'b', 'u', 'v'], ('g', 'i'): ['g', 'b', 'u', 'v', 'i'], ('g', 'n'): ['g', 'b', 'u', 'v', 'i', 'n'], ('u', 'a'): ['u', 'b', 'p', 'r', 's', 'a'], ('u', 'z'): ['u', 'b', 'p', 'r', 's', 'a', 'z'], ('u', 'o'): ['u', 'b', 'p', 'r', 's', 'o'], ('u', 's'): ['u', 'b', 'p', 'r', 's'], ('u', 't'): ['u', 'b', 'p', 'r', 's', 'a', 't'], ('u', 'l'): ['u', 'b', 'p', 'c', 'd', 'm', 'l'], ('u', 'm'): ['u', 'b', 'p', 'c', 'd', 'm'], ('u', 'd'): ['u', 'b', 'p', 'c', 'd'], ('u', 'c'): ['u', 'b', 'p', 'c'], ('u', 'r'): ['u', 'b', 'p', 'r'], ('u', 'p'): ['u', 'b', 'p'], ('u', 'b'): ['u', 'b'], ('u', 'f'): ['u', 'b', 'f'], ('u', 'g'): ['u', 'b', 'g'], ('u', 'u'): [], ('u', 'h'): ['u', 'h'], ('u', 'e'): ['u', 'h', 'e'], ('u', 'v'): ['u', 'v'], ('u', 'i'): ['u', 'v', 'i'], ('u', 'n'): ['u', 'v', 'i', 'n'], ('h', 'a'): ['h', 'u', 'b', 'p', 'r', 's', 'a'], ('h', 'z'): ['h', 'u', 'b', 'p', 'r', 's', 'a', 'z'], ('h', 'o'): ['h', 'u', 'b', 'p', 'r', 's', 'o'], ('h', 's'): ['h', 'u', 'b', 'p', 'r', 's'], ('h', 't'): ['h', 'u', 'b', 'p', 'r', 's', 'a', 't'], ('h', 'l'): ['h', 'u', 'b', 'p', 'c', 'd', 'm', 'l'], ('h', 'm'): ['h', 'u', 'b', 'p', 'c', 'd', 'm'], ('h', 'd'): ['h', 'u', 'b', 'p', 'c', 'd'], ('h', 'c'): ['h', 'u', 'b', 'p', 'c'], ('h', 'r'): ['h', 'u', 'b', 'p', 'r'], ('h', 'p'): ['h', 'u', 'b', 'p'], ('h', 'b'): ['h', 'u', 'b'], ('h', 'f'): ['h', 'u', 'b', 'f'], ('h', 'g'): ['h', 'u', 'b', 'g'], ('h', 'u'): ['h', 'u'], ('h', 'h'): [], ('h', 'e'): ['h', 'e'], ('h', 'v'): ['h', 'u', 'v'], ('h', 'i'): ['h', 'u', 'v', 'i'], ('h', 'n'): ['h', 'u', 'v', 'i', 'n'], ('e', 'a'): ['e', 'h', 'u', 'b', 'p', 'r', 's', 'a'], ('e', 'z'): ['e', 'h', 'u', 'b', 'p', 'r', 's', 'a', 'z'], ('e', 'o'): ['e', 'h', 'u', 'b', 'p', 'r', 's', 'o'], ('e', 's'): ['e', 'h', 'u', 'b', 'p', 'r', 's'], ('e', 't'): ['e', 'h', 'u', 'b', 'p', 'r', 's', 'a', 't'], ('e', 'l'): ['e', 'h', 'u', 'b', 'p', 'c', 'd', 'm', 'l'], ('e', 'm'): ['e', 'h', 'u', 'b', 'p', 'c', 'd', 'm'], ('e', 'd'): ['e', 'h', 'u', 'b', 'p', 'c', 'd'], ('e', 'c'): ['e', 'h', 'u', 'b', 'p', 'c'], ('e', 'r'): ['e', 'h', 'u', 'b', 'p', 'r'], ('e', 'p'): ['e', 'h', 'u', 'b', 'p'], ('e', 'b'): ['e', 'h', 'u', 'b'], ('e', 'f'): ['e', 'h', 'u', 'b', 'f'], ('e', 'g'): ['e', 'h', 'u', 'b', 'g'], ('e', 'u'): ['e', 'h', 'u'], ('e', 'h'): ['e', 'h'], ('e', 'e'): [], ('e', 'v'): ['e', 'h', 'u', 'v'], ('e', 'i'): ['e', 'h', 'u', 'v', 'i'], ('e', 'n'): ['e', 'h', 'u', 'v', 'i', 'n'], ('v', 'a'): ['v', 'u', 'b', 'p', 'r', 's', 'a'], ('v', 'z'): ['v', 'u', 'b', 'p', 'r', 's', 'a', 'z'], ('v', 'o'): ['v', 'u', 'b', 'p', 'r', 's', 'o'], ('v', 's'): ['v', 'u', 'b', 'p', 'r', 's'], ('v', 't'): ['v', 'u', 'b', 'p', 'r', 's', 'a', 't'], ('v', 'l'): ['v', 'u', 'b', 'p', 'c', 'd', 'm', 'l'], ('v', 'm'): ['v', 'u', 'b', 'p', 'c', 'd', 'm'], ('v', 'd'): ['v', 'u', 'b', 'p', 'c', 'd'], ('v', 'c'): ['v', 'u', 'b', 'p', 'c'], ('v', 'r'): ['v', 'u', 'b', 'p', 'r'], ('v', 'p'): ['v', 'u', 'b', 'p'], ('v', 'b'): ['v', 'u', 'b'], ('v', 'f'): ['v', 'u', 'b', 'f'], ('v', 'g'): ['v', 'u', 'b', 'g'], ('v', 'u'): ['v', 'u'], ('v', 'h'): ['v', 'u', 'h'], ('v', 'e'): ['v', 'u', 'h', 'e'], ('v', 'v'): [], ('v', 'i'): ['v', 'i'], ('v', 'n'): ['v', 'i', 'n'], ('i', 'a'): ['i', 'v', 'u', 'b', 'p', 'r', 's', 'a'], ('i', 'z'): ['i', 'v', 'u', 'b', 'p', 'r', 's', 'a', 'z'], ('i', 'o'): ['i', 'v', 'u', 'b', 'p', 'r', 's', 'o'], ('i', 's'): ['i', 'v', 'u', 'b', 'p', 'r', 's'], ('i', 't'): ['i', 'v', 'u', 'b', 'p', 'r', 's', 'a', 't'], ('i', 'l'): ['i', 'v', 'u', 'b', 'p', 'c', 'd', 'm', 'l'], ('i', 'm'): ['i', 'v', 'u', 'b', 'p', 'c', 'd', 'm'], ('i', 'd'): ['i', 'v', 'u', 'b', 'p', 'c', 'd'], ('i', 'c'): ['i', 'v', 'u', 'b', 'p', 'c'], ('i', 'r'): ['i', 'v', 'u', 'b', 'p', 'r'], ('i', 'p'): ['i', 'v', 'u', 'b', 'p'], ('i', 'b'): ['i', 'v', 'u', 'b'], ('i', 'f'): ['i', 'v', 'u', 'b', 'f'], ('i', 'g'): ['i', 'v', 'u', 'b', 'g'], ('i', 'u'): ['i', 'v', 'u'], ('i', 'h'): ['i', 'v', 'u', 'h'], ('i', 'e'): ['i', 'v', 'u', 'h', 'e'], ('i', 'v'): ['i', 'v'], ('i', 'i'): [], ('i', 'n'): ['i', 'n'], ('n', 'a'): ['n', 'i', 'v', 'u', 'b', 'p', 'r', 's', 'a'], ('n', 'z'): ['n', 'i', 'v', 'u', 'b', 'p', 'r', 's', 'a', 'z'], ('n', 'o'): ['n', 'i', 'v', 'u', 'b', 'p', 'r', 's', 'o'], ('n', 's'): ['n', 'i', 'v', 'u', 'b', 'p', 'r', 's'], ('n', 't'): ['n', 'i', 'v', 'u', 'b', 'p', 'r', 's', 'a', 't'], ('n', 'l'): ['n', 'i', 'v', 'u', 'b', 'p', 'c', 'd', 'm', 'l'], ('n', 'm'): ['n', 'i', 'v', 'u', 'b', 'p', 'c', 'd', 'm'], ('n', 'd'): ['n', 'i', 'v', 'u', 'b', 'p', 'c', 'd'], ('n', 'c'): ['n', 'i', 'v', 'u', 'b', 'p', 'c'], ('n', 'r'): ['n', 'i', 'v', 'u', 'b', 'p', 'r'], ('n', 'p'): ['n', 'i', 'v', 'u', 'b', 'p'], ('n', 'b'): ['n', 'i', 'v', 'u', 'b'], ('n', 'f'): ['n', 'i', 'v', 'u', 'b', 'f'], ('n', 'g'): ['n', 'i', 'v', 'u', 'b', 'g'], ('n', 'u'): ['n', 'i', 'v', 'u'], ('n', 'h'): ['n', 'i', 'v', 'u', 'h'], ('n', 'e'): ['n', 'i', 'v', 'u', 'h', 'e'], ('n', 'v'): ['n', 'i', 'v'], ('n', 'i'): ['n', 'i'], ('n', 'n'): []}
        
        
        start = 's' 
        goal = 'z'
        
        nodes = ['a','z','o','s','t','l','m','d','c','r','p','b','f','g','u','h','e','v','i','n']
        # for start in nodes:
        #     # counter = 0
        #     for goal in nodes[nodes.index(start):]:
        
            

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}
    
        self.romania.reset_search()
        path = bidirectional_ucs(self.romania, start, goal)
        # print(path)
        if correct_tests_ucs[(start,goal)] != path:
            print(start,' to ',goal,' error')
        
        # if len(path)!=0:
        #     counter+=1
        # if goal == 'n':
        #     print(start, ' ', counter)
    
        # TestBasicSearch.draw_graph(self.romania, node_positions=node_positions,
        #                 start=start, goal=goal, path=path,
        #                 title='bi-ucs blue=start, yellow=goal, green=explored')

    

    def test_bidirectional_ucs_explored(self):
        """Test Bi-UCS for correct path and number of explored nodes"""
        start = 'o'
        goal = 'd'

        node_positions = {n: self.romania.nodes[n]['pos'] for n in
                          self.romania.nodes.keys()}

        self.romania.reset_search()
        path = bidirectional_ucs(self.romania, start, goal)

        self.assertEqual(path, ['o', 's', 'r', 'c', 'd'])   # Check for correct path. Check your stopping condition

        explored_nodes = sum(list(self.romania.explored_nodes().values()))
        # print('BiUCS explore', explored_nodes, list(self.romania.explored_nodes.values()))
        self.assertLessEqual(explored_nodes, 12)    # Compare explored nodes to reference implementation

    def test_bidirectional_a_star(self):
        """Test and generate GeoJSON for bidirectional A* search"""
        path = bidirectional_a_star(self.atlanta, '69581003', '69581000', heuristic=haversine_dist_heuristic)
        all_explored = self.atlanta.explored_nodes()
        plot_search(self.atlanta, 'atlanta_search_bidir_a_star.json', path,
                    all_explored)

if __name__ == '__main__':
    unittest.main()
