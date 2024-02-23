import copy

from alphago import agents, rl
from alphago.encoders.one_plane import OnePlaneEncoder
from alphago.env.go_board import GameState, Player, Point
from alphago.env.gotypes import Player
from alphago.env.scoring import compute_game_result
from alphago.networks import GoNet


def simulate_game(black_player, white_player, board_size=19):
    moves = []
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    game_result = compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def main():
    num_games = 1
    # agent1 = agents.MCTSAgent(1000, 0.8)
    # agent2 = agents.MCTSAgent(1000, 0.8)
    board_size = 5
    agent1 = agents.PGAgent(
        GoNet(board_size), OnePlaneEncoder((board_size, board_size))
    )
    agent2 = agents.PGAgent(
        GoNet(board_size), OnePlaneEncoder((board_size, board_size))
    )
    collector1 = rl.ExperienceCollector()
    collector2 = rl.ExperienceCollector()
    agent1.set_buffer(collector1)
    agent2.set_buffer(collector2)

    for i in range(num_games):
        print(f"---- Game {i+1} -----")
        collector1.begin_episode()
        collector2.begin_episode()
        records = simulate_game(agent1, agent2)
        if records.winner == Player.black:
            collector1.store_episode(reward=1)
            collector2.store_episode(reward=-1)
        else:
            collector1.store_episode(reward=1)
            collector2.store_episode(reward=-1)

    experience = rl.ExperienceBuffer.combine_buffers([collector1, collector2])
    experience.save_exp('experiment/experience/exp1.npz')


if __name__ == "__main__":
    main()
