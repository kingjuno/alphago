from alphago import rl
from alphago.env import go_board, scoring
from alphago.env.gotypes import Player


class GameRecord:
    def __init__(self, moves, winner, margin):
        self.moves = moves
        self.winner = winner
        self.margin = margin


def simulate_game(black_player, white_player):
    moves = []
    game = go_board.GameState.new_game(19)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)
    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def experience_simulation(num_games, agent1, agent2):
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()

    color1 = Player.black
    for i in range(num_games):
        collector1.begin_episode()
        agent1.set_buffer(collector1)
        collector2.begin_episode()
        agent2.set_buffer(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent2, agent1
        print(f"Game {i+1}: ", end="")
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        color1 = color1.other

    return rl.ExperienceBuffer.combine_buffers([collector1, collector2])
