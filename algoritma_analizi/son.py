from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import math, heapq, random, sys
from typing import Dict, Set, Tuple, List, Optional
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

class GraphVisualizer:
    def __init__(self, node_count: int = 100, edge_count: int = 300):
        self.node_count = node_count
        self.edge_count = edge_count
        self.graph = self._generate_graph()
        self.pos = self._generate_positions()
        
    def _generate_positions(self) -> Dict[int, Tuple[float, float]]:
        #dugum konumlarini daire seklinde olusturma
        positions = {}
        for i in range(self.node_count):
            angle = 2 * math.pi * i / self.node_count
            x = math.cos(angle)
            y = math.sin(angle)
            positions[i] = (x, y)
        return positions
    
    def _generate_graph(self) -> Dict[int, Dict[int, int]]:
        #rastgele agırlıklı graf olusturma
        graph = {i: {} for i in range(self.node_count)}
        added_edges = set()
        
        while len(added_edges) < self.edge_count:
            u = random.randint(0, self.node_count - 1)
            v = random.randint(0, self.node_count - 1)
            if u != v and (u, v) not in added_edges and (v, u) not in added_edges:
                weight = random.randint(1, 20)
                graph[u][v] = weight
                graph[v][u] = weight
                added_edges.add((u, v))
            
        return graph

class DijkstraAlgorithm:
    def __init__(self, graph: Dict[int, Dict[int, int]]):
        self.graph = graph
        self.distances: Dict[int, float] = {}
        self.previous: Dict[int, Optional[int]] = {}
        self.visited: Set[int] = set()
    
    def find_shortest_path(self, start: int, target: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        #Dijkstra algoritmasını calismasi 
        self.distances = {node: math.inf for node in self.graph}
        self.distances[start] = 0
        self.previous = {node: None for node in self.graph}
        self.visited = set()
        
        heap = []
        heapq.heappush(heap, (0, start))
        
        while heap:
            dist, current = heapq.heappop(heap)
            
            if current in self.visited:
                continue
                
            self.visited.add(current)
            
            if current == target:
                break
                
            for neighbor, weight in self.graph[current].items():
                if neighbor in self.visited:
                    continue
                    
                new_dist = dist + weight
                if new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.previous[neighbor] = current
                    heapq.heappush(heap, (new_dist, neighbor))
        
        return self.distances, self.previous
    
    @staticmethod
    def reconstruct_path(previous: Dict[int, Optional[int]], start: int, target: int) -> List[int]:
        #en kısa yolu onceki dugumden olusturma
        path = []
        node = target
        
        while node != start:
            if node not in previous:
                return []  # yol yok
            path.append(node)
            node = previous[node]
        
        path.append(start)
        return list(reversed(path))

class DijkstraRunner(QtCore.QThread):
    update_visual = QtCore.pyqtSignal(object, str, object)

    def __init__(self, graph_visualizer, start, target, delay=1.0):
        super().__init__()
        self.gv = graph_visualizer
        self.start_node = start
        self.target = target
        self.delay = delay
        self.dijkstra = DijkstraAlgorithm(self.gv.graph)
    
    def get_current_tree_edges(self) -> Set[Tuple[int, int]]:
        tree_edges = set()
        for node, prev in self.dijkstra.previous.items():
            if prev is not None:
                tree_edges.add((min(node, prev), max(node, prev)))
        return tree_edges

    def run(self):
        self.dijkstra.distances = {node: math.inf for node in self.gv.graph}
        self.dijkstra.distances[self.start_node] = 0
        self.dijkstra.previous = {node: None for node in self.gv.graph}
        self.dijkstra.visited = set()
        
        heap = []
        heapq.heappush(heap, (0, self.start_node))

        current = self.start_node  

        tree_edges = self.get_current_tree_edges()
        self.update_visual.emit(
            (current, self.dijkstra.visited.copy(), tree_edges, self.start_node, self.target),
            "Algoritma başlatildi...",
            (self.gv.graph, set())  
        )

        self.msleep(int(self.delay * 1000))

        while heap:
            dist, current = heapq.heappop(heap)
            if current in self.dijkstra.visited:
                continue

            self.dijkstra.visited.add(current)

            neighbors = ', '.join(str(n) for n in self.gv.graph[current])
            tree_edges = self.get_current_tree_edges()
            self.update_visual.emit(
                (current, self.dijkstra.visited.copy(), tree_edges, self.start_node, self.target), 
                f"Komşular inceleniyor: {neighbors}",
                (self.gv.graph, set())  
            )

            self.msleep(int(self.delay * 1000))

            if current == self.target:
                break

            for neighbor, weight in self.gv.graph[current].items():
                if neighbor in self.dijkstra.visited:
                    continue

                new_dist = dist + weight
                if new_dist < self.dijkstra.distances[neighbor]:
                    old_dist = self.dijkstra.distances[neighbor]
                    self.dijkstra.distances[neighbor] = new_dist
                    self.dijkstra.previous[neighbor] = current
                    heapq.heappush(heap, (new_dist, neighbor))

                    tree_edges = self.get_current_tree_edges()
                    self.update_visual.emit(
                        (current, self.dijkstra.visited.copy(), tree_edges, self.start_node, self.target), 
                        f"Yol güncellendi: {current} → {neighbor} (eski: {old_dist}, yeni: {new_dist})",
                        (self.gv.graph, set())  
                    )

                    self.msleep(int(self.delay * 1000))

        # sonuc yolu
        path = DijkstraAlgorithm.reconstruct_path(self.dijkstra.previous, self.start_node, self.target)
        path_edges = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            path_edges.add((min(u, v), max(u, v)))

        tree_edges = self.get_current_tree_edges()
       
        # yol aciklaması
        path_description = " -> ".join(str(node) for node in path)
        shortest_path_text = f"En kısa yol: {path_description}"

        # toplam yol
        weights = []
        total_weight = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            w = self.gv.graph[u][v]
            weights.append(str(w))
            total_weight += w

        weights_str = " + ".join(weights) + f" = {total_weight}"
        distance_text = f"Toplam mesafe: {weights_str}"

        self.update_visual.emit(
            (current, self.dijkstra.visited.copy(), tree_edges, self.start_node, self.target),
            f"{shortest_path_text}\n{distance_text}",
            (self.gv.graph, path_edges)
        )

class DijkstraWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijkstra Algoritmasının Uygulanması")
        self.setGeometry(100, 100, 900, 800)

        # graf olusturma
        self.gv = GraphVisualizer(300, 600)
        
        # arayuz
        self.canvas = FigureCanvas(plt.figure(figsize=(8, 8)))
        self.text_panel = QtWidgets.QTextEdit()
        self.text_panel.setReadOnly(True)
        self.text_panel.setMaximumHeight(150)
        
        # butonları
        self.start_btn = QtWidgets.QPushButton("Başlat ( Başlangıç Node:0 →Hedef Node: 150)")
        self.start_btn.clicked.connect(self.start_algorithm)
        
        # Layout
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(self.start_btn)
        
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.text_panel)
        
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def start_algorithm(self):
        self.text_panel.clear()
        #  once sadece agirliklariyla  grafı göster 
        self.update_visualization(
            state_info=(None, set(), set(), 0, 150),  
            text="Graf oluşturuldu. 5 saniye sonra algoritma başlayacak...",
            graph_info=(self.gv.graph, set()) 
        )
         #  5 saniye sonra algoritma calismaya baslar
        QtCore.QTimer.singleShot(5000, self._start_dijkstra_algorithm)  

    def _start_dijkstra_algorithm(self):
        self.runner = DijkstraRunner(self.gv, start=0, target=150, delay=0.5)
        self.runner.update_visual.connect(self.update_visualization)
        self.runner.start()

    def update_visualization(self, state_info, text, graph_info):
        graph, path_edges= graph_info
        current, visited, tree_edges, start_node, target_node = state_info
        
        is_initial_draw = len(visited) == 0 and not path_edges 

        self.text_panel.append(text)

        ax = self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.axis('off')
        
        # final path cizgilerini kirmizi
        # Tree edges (anlik yollar) mavi cizgiyle cizme
        # kenarlari ciz
        for u in graph:
            for v in graph[u]:
                if u < v:
                    edge = (u, v)
                    color = "gray"
                    width = 0.5
                    draw_weight = False

                    # final yol gosterme
                    if path_edges and len(path_edges)>0:
                        if edge in path_edges:
                            color = "red"
                            width = 2
                            draw_weight = True  # final yolun agirlik yazma
                        else:
                            continue
                        
                    elif is_initial_draw:
                        draw_weight = True  # baslangicta tum agirliklar yazilmasi
                      
                    elif tree_edges:
                        # algoritma sirasinda gecici yollar
                        if edge in tree_edges:
                            color = "blue"
                            width = 1.5
                        else:
                            color = "gray"
                            width = 0.5

                    ax.plot([self.gv.pos[u][0], self.gv.pos[v][0]],
                            [self.gv.pos[u][1], self.gv.pos[v][1]],
                            color=color, linewidth=width, alpha=0.6)

                    if draw_weight:
                        mx = (self.gv.pos[u][0] + self.gv.pos[v][0]) / 2
                        my = (self.gv.pos[u][1] + self.gv.pos[v][1]) / 2
                        weight = graph[u][v]
                        ax.text(mx, my, str(weight), fontsize=6, ha='center', va='center', color='black')

        # dugum cizme
        for node in graph:
            if node == start_node:
                color = "green"
                size = 80
            elif node == target_node:
                color = "blue"
                size = 80
            elif node == current:
                color = "orange"
                size = 60
            elif node in visited:
                color = "lightgreen"
                size = 40
            else:
                color = "white"
                size = 30
                
            ax.scatter(self.gv.pos[node][0], self.gv.pos[node][1], 
                      c=color, s=size, edgecolors='black', linewidths=0.5)
        
        # dugum label'larinin yazilmasi
        for node in graph:
            x, y = self.gv.pos[node]
            if node % 10 == 0:  # 10'un katları
                offset = 0.02  # düğümden uzaklığı
                vertical_offset = offset  
                
                # 120-180 veya 180-270 arası düğümlerin label'ı aşağı yaz
                if 120 <= node <= 180 or 180 < node <= 270:
                    vertical_offset = -offset 
                    
                # 110-149 arası düğümlerin etiketini sola, diğerlerini sağa yaz
                if 110 <= node <= 149:
                    ax.text(x - offset, y + vertical_offset, str(node), 
                            fontsize=8, style='italic', ha='right', va='center', color='red')
                else:
                    ax.text(x + offset, y + vertical_offset, str(node), 
                            fontsize=8, style='italic', ha='left', va='center', color='red')
                    
        # baslik ve aciklama
        progress = f"{len(visited)}/{self.gv.node_count} ({len(visited)/self.gv.node_count:.1%})"
        title = f"Current: {current}" if current is not None else "Algorithm Completed"
        ax.set_title(f"{title} - Visited: {progress}")
    
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Start', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Target', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Current', markerfacecolor='orange', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Visited', markerfacecolor='lightgreen', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Unvisited', markerfacecolor='white', markersize=8),
            Line2D([0], [0], color='blue', lw=2, label='Geçici yollar'),
            Line2D([0], [0], color='red', lw=2, label='Final yol')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.13, 1.15), fontsize=8)

        self.canvas.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DijkstraWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()