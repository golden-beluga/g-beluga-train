import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

def tansho(tickets, target_ranks, prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket][0]
        if rank == 1:
            num_hit += 1
            ret += prize
            
    return num_hit, ret

def fukusho(tickets, target_ranks, first_prize, second_prize, third_prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket][0]
        if rank == 1:
            num_hit += 1
            ret += first_prize 
        elif rank == 2:
            num_hit += 1
            ret += second_prize 
        elif rank == 3 and len(target_ranks) > 7:
            num_hit += 1
            ret += third_prize             

    return num_hit, ret

def wakuren(tickets, target_ranks, prize):
    raise NotImplementedError()

def umaren(tickets, target_ranks, prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket]
        if rank[0] in [1, 2] and rank[1] in [1, 2]:
            num_hit += 1
            ret += prize 

    return num_hit, ret

def umatan(tickets, target_ranks, prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket]
        if rank[0] == 1 and rank[1] == 2:
            num_hit += 1
            ret += prize 

    return num_hit, ret

def wide(tickets, target_ranks, first_second_prize, first_third_prize, second_third_prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket]

        # 1着, 2着
        if (rank[0] == 1 and rank[1] == 2) or (rank[0] == 2 and rank[1] == 1):
            num_hit += 1
            ret += first_second_prize
        # 1着, 3着
        elif (rank[0] == 1 and rank[1] == 3) or (rank[0] == 3 and rank[1] == 1):
            num_hit += 1
            ret += first_third_prize
        # 2着, 3着
        elif (rank[0] == 2 and rank[1] == 3) or (rank[0] == 3 and rank[1] == 2):
            num_hit += 1
            ret += second_third_prize

    return num_hit, ret

def sanrempuku(tickets, target_ranks, prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket]
        if rank[0] in [1, 2, 3] and rank[1] in [1, 2, 3] and rank[2] in [1, 2, 3]:
            num_hit += 1
            ret += prize 

    return num_hit, ret

def sanrentan(tickets, target_ranks, prize):
    num_hit = 0
    ret = 0
    for ticket in tickets:
        rank = target_ranks[ticket]
        if rank[0] == 1 and rank[1] == 2 and rank[2] == 3:
            num_hit += 1
            ret += prize 

    return num_hit, ret

if __name__ == "__main__":
    from utils.top_n_box import TopNBox
    predicted_ranks = np.array([1, 3, 4, 2, 5, 6])
    target_ranks = np.array([1, 2, 3, 4, 5, 6])
    prize = 500
    top_n_box = TopNBox(predicted_ranks, 4)
    for t in ["sanrentan"]:#TopNBox.ticket_types:
        tickets = top_n_box.output_tickets(t)
        num_hit, ret = sanrentan(tickets, target_ranks, prize)
        print(f"-----ticket type: {t}------")
        print("predicted ranks:", predicted_ranks)
        print("prize:", prize)
        print("tickets: ", tickets)
        print("num hit:", num_hit)
        print("return:", ret)
