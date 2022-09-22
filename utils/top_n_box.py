import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import itertools
import numpy as np
from utils.purchasing_strategy import PurchasingStrategy

class TopNBox(PurchasingStrategy):
    def __init__(self, predicted_ranks, n):
        super().__init__(predicted_ranks)
        self.n = n

    def tansho(self):
        tickets = []
        for i in range(1, self.n+1):
            ticket = np.where(self.predicted_ranks == i)
            # ticket => (array([something...]), ) 
            # ticket[0] => array[something...]
            tickets.append(ticket[0])
        return tickets

    def fukusho(self):
        tickets = self.tansho()
        return tickets

    def wakuren(self):
        return []

    def umaren(self):
        if self.n < 2:
            raise "Error: n must be bigger than 1."
        top_n = []
        for i in range(1, self.n+1):
            index = np.where(self.predicted_ranks == i)
            # index => (array([something...]), ) 
            # index[0] => array[something...]
            # index[0][0] => something
            top_n.append(index[0][0])
        tickets = []
        for ticket in itertools.combinations(top_n, 2):
            tickets.append(np.array(ticket))

        return tickets

    def umatan(self):
        if self.n < 2:
            raise "Error: n must be bigger than 1."
        top_n = []
        for i in range(1, self.n+1):
            index = np.where(self.predicted_ranks == i)
            # index => (array([something...]), ) 
            # index[0] => array[something...]
            # index[0][0] => something
            top_n.append(index[0][0])
        tickets = []
        for ticket in itertools.permutations(top_n, 2):
            tickets.append(np.array(ticket))

        return tickets

    def wide(self):
        tickets = self.umaren()
        return tickets

    def sanrempuku(self):
        if self.n < 3:
            raise "Error: n must be bigger than 2."
        top_n = []
        for i in range(1, self.n+1):
            index = np.where(self.predicted_ranks == i)
            # index => (array([something...]), ) 
            # index[0] => array[something...]
            # index[0][0] => something
            top_n.append(index[0][0])
        tickets = []
        for ticket in itertools.combinations(top_n, 3):
            tickets.append(np.array(ticket))

        return tickets

    def sanrentan(self):
        if self.n < 3:
            raise "Error: n must be bigger than 2."
        top_n = []
        for i in range(1, self.n+1):
            index = np.where(self.predicted_ranks == i)
            # index => (array([something...]), ) 
            # index[0] => array[something...]
            # index[0][0] => something
            top_n.append(index[0][0])
        tickets = []
        for ticket in itertools.permutations(top_n, 3):
            tickets.append(np.array(ticket))

        return tickets

if __name__ == "__main__":
    predicted_ranks = np.array([1, 3, 4, 2, 5, 6])
    top_n_box = TopNBox(predicted_ranks, 2)
    for t in TopNBox.ticket_types:
        print(f"-----ticket type: {t}------")
        print("predicted ranks:", predicted_ranks)
        print("tickets: ", top_n_box.output_tickets(t))
