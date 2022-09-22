def correct_rate(target_ranks: list, predicted_ranks: list) -> float:
    correct_num = 0
    for t_r, p_r in zip(target_ranks, predicted_ranks):
        if target_ranks == predicted_ranks:
            correct_num += 1
            
    rate = correct_num / len(target_ranks)
    
    return rate