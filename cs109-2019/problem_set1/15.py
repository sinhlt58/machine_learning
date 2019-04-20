import random

def simulate_a_game():
    S = 0
    x = 0
    y = 0
    while S <= 100:
        x = random.randint(1, 100)
        S += x
    while S <= 200:
        y = random.randint(1, 100)
        S += y
    return y - x

if __name__ == "__main__":
    num_y_won = 0
    M = 100000
    for i in range(0, M):
        r = simulate_a_game()
        if r > 0:
            num_y_won += 1

    print ('Probability y wins is {:0.2f}'.format(num_y_won / M))