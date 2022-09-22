from abc import *

class PurchasingStrategy:
    ticket_types = [
        "tansho", 
        "fukusho", 
        "umaren", 
        "umatan", 
        "wide", 
        "sanrempuku", 
        "sanrentan"]
        
    def __init__(self, predicted_ranks):
        self.predicted_ranks = predicted_ranks

    def output_tickets(self, ticket_type):
        tickets = []
        if ticket_type == self.ticket_types[0]:
            tickets = self.tansho()
        elif ticket_type == self.ticket_types[1]:
            tickets = self.fukusho()
        elif ticket_type == self.ticket_types[2]:
            tickets = self.umaren()
        elif ticket_type == self.ticket_types[3]:
            tickets = self.umatan()
        elif ticket_type == self.ticket_types[4]:
            tickets = self.wide()
        elif ticket_type == self.ticket_types[5]:
            tickets = self.sanrempuku()
        elif ticket_type == self.ticket_types[6]:
            tickets = self.sanrentan()
        
        return tickets
    
    @abstractmethod
    def tansho(self):
        raise NotImplementedError()

    @abstractmethod
    def fukusho(self):
        raise NotImplementedError()

    @abstractmethod
    def wakuren(self):
        raise NotImplementedError()

    @abstractmethod
    def umaren(self):
        raise NotImplementedError()

    @abstractmethod
    def umatan(self):
        raise NotImplementedError()

    @abstractmethod
    def wide(self):
        raise NotImplementedError()

    @abstractmethod
    def sanrempuku(self):
        raise NotImplementedError()

    @abstractmethod
    def sanrentan(self):
        raise NotImplementedError()
